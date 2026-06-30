#ifndef TRANSPORTSTEP_HPP
#define TRANSPORTSTEP_HPP

#include "VirtualSolver.hpp"
#include "Reconstruction.hpp"
#include "Gradient.hpp"
#include <type_traits>

// SFINAE trait: detect if Model<T> has a static diffusive_flux method
template <typename T, typename = void>
struct HasDiffusiveFlux : std::false_type {};

template <typename T>
struct HasDiffusiveFlux<T, std::void_t<decltype(
    Model<T>::diffusive_flux(
        std::declval<const T*>(), std::declval<const T*>(),
        std::declval<const T*>(), std::declval<const T*>())
)>> : std::true_type {};

// SFINAE: does Model<T> emit update_variables (the pointwise state post-update /
// wet-dry momentum clamp)? Not every model has it (plain SME does not).
template <typename T, typename = void>
struct HasUpdateVariables : std::false_type {};
template <typename T>
struct HasUpdateVariables<T, std::void_t<decltype(
    Model<T>::update_variables(
        std::declval<const T*>(), std::declval<const T*>(),
        std::declval<const T*>(), std::declval<const T>()))>>
    : std::true_type {};

// SFINAE: does Model<T> have the field-level, mesh-aware update_aux_variables
// (REQ-65) that refreshes the spatial-derivative aux via compute_derivative?
template <typename T, typename = void>
struct HasFieldAuxUpdate : std::false_type {};
template <typename T>
struct HasFieldAuxUpdate<T, std::void_t<decltype(
    Model<T>::update_aux_variables(
        std::declval<T* const*>(), std::declval<T* const*>(),
        std::declval<const T>(), std::declval<const ZoomyMesh&>()))>>
    : std::true_type {};

template <typename T>
class TransportStep {
private:
    DM dmQ, dmAux, dmGrad; 
    std::shared_ptr<Reconstructor<T>> reconstructor;
    std::shared_ptr<Reconstructor<T>> reconstructorAux; 
    std::shared_ptr<GradientCalculator<T>> gradient;
    
    std::vector<T> parameters;
    std::map<PetscInt, PetscInt> boundary_map;
    
    Vec V_min_loc, V_max_loc; 
    Vec A_min_loc, A_max_loc;

    std::vector<PetscInt> neigh_offsets, neigh_list;
    bool topology_setup = false;

    std::vector<uint8_t> cell_orders;
    // Local-MOOD face filter: when set, ComputeFluxes processes ONLY faces
    // adjacent to a troubled cell (mask[c-cStart]==1) and accumulates into F for
    // those cells. Used to compute a cheap, local O1 override of troubled cells.
    const std::vector<uint8_t>* troubled_filter = nullptr;

    FluxKernelPtr cons_flux_kernel;
    NonConservativeFluxKernelPtr noncons_flux_kernel;
    SourceKernelPtr source_kernel;
    bool do_refresh_deriv_aux = true;  // gate the per-step mesh-derivative aux refresh
    bool wb_positivity = false;        // order-2 eta-WB + Zhang-Shu positivity limiting
    bool mood_skip_theta = false;      // MOOD mode: eta-WB + Venkat but NO Zhang-Shu theta
    T wet_dry_eps = (T)1e-2;

    bool IsOwned(DM dm, PetscInt p) {
        PetscInt g_idx; 
        DMPlexGetPointGlobal(dm, p, &g_idx, NULL); 
        return (g_idx >= 0);
    }

public:
    TransportStep(DM q, DM aux, DM grad, std::vector<T> params, std::map<PetscInt, PetscInt> bcs) 
        : dmQ(q), dmAux(aux), dmGrad(grad), parameters(params), boundary_map(bcs) {
        V_min_loc = NULL; V_max_loc = NULL;
        A_min_loc = NULL; A_max_loc = NULL;
        reconstructor = std::make_shared<PCMReconstructor<T>>(); 
        reconstructorAux = std::make_shared<PCMReconstructor<T>>(Model<T>::n_dof_qaux); 
        gradient = nullptr;
        cons_flux_kernel = nullptr; 
        noncons_flux_kernel = nullptr; 
        source_kernel = Model<T>::source;
    }

    ~TransportStep() {
        if (V_min_loc) VecDestroy(&V_min_loc);
        if (V_max_loc) VecDestroy(&V_max_loc);
        if (A_min_loc) VecDestroy(&A_min_loc);
        if (A_max_loc) VecDestroy(&A_max_loc);
    }

    void SetReconstruction(std::shared_ptr<Reconstructor<T>> r) { reconstructor = r; }
    void SetAuxReconstruction(std::shared_ptr<Reconstructor<T>> r) { reconstructorAux = r; }
    void SetGradient(std::shared_ptr<GradientCalculator<T>> g) { gradient = g; }
    void SetNonConsFlux(NonConservativeFluxKernelPtr k) { noncons_flux_kernel = k; }
    void SetFluxKernel(FluxKernelPtr k) { cons_flux_kernel = k; }
    void SetCellOrders(const std::vector<uint8_t>& orders) { cell_orders = orders; }

    PetscErrorCode SetupTopology() {
        if (topology_setup) return PETSC_SUCCESS;
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        neigh_offsets.clear(); neigh_list.clear(); neigh_offsets.push_back(0);
        for(PetscInt c=cStart; c<cEnd; ++c) {
            PetscInt num_adj = -1; PetscInt *adj = NULL; PetscCall(DMPlexGetAdjacency(dmQ, c, &num_adj, &adj));
            for(int k=0; k<num_adj; ++k) { PetscInt n = adj[k]; if (n == c || n < 0) continue; neigh_list.push_back(n); }
            PetscCall(PetscFree(adj)); neigh_offsets.push_back(neigh_list.size());
        }
        PetscCall(DMCreateLocalVector(dmQ, &V_min_loc)); 
        PetscCall(DMCreateLocalVector(dmQ, &V_max_loc));
        PetscCall(DMCreateLocalVector(dmAux, &A_min_loc)); 
        PetscCall(DMCreateLocalVector(dmAux, &A_max_loc));
        topology_setup = true; return PETSC_SUCCESS;
    }

    PetscErrorCode UpdateNeighborBounds(Vec X_loc, Vec A_loc) {
        if (!topology_setup) PetscCall(SetupTopology());
        const PetscScalar *x_ptr; PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscScalar *min_ptr, *max_ptr;
        PetscCall(VecGetArray(V_min_loc, &min_ptr)); PetscCall(VecGetArray(V_max_loc, &max_ptr));
        PetscInt cStart, cEnd; 
        PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        
        for (PetscInt c = cStart; c < cEnd; ++c) {
            const PetscScalar *q; PetscCall(DMPlexPointLocalRead(dmQ, c, x_ptr, &q));
            PetscScalar q_min[Model<T>::n_dof_q], q_max[Model<T>::n_dof_q];
            for(int i=0; i<Model<T>::n_dof_q; ++i) { q_min[i] = q[i]; q_max[i] = q[i]; }

            int idx_start = neigh_offsets[c - cStart]; 
            int idx_end = neigh_offsets[c - cStart + 1];
            
            for(int k=idx_start; k<idx_end; ++k) {
                PetscInt n = neigh_list[k]; 
                const PetscScalar *qn; PetscCall(DMPlexPointLocalRead(dmQ, n, x_ptr, &qn));
                for(int i=0; i<Model<T>::n_dof_q; ++i) { 
                    if (qn[i] < q_min[i]) q_min[i] = qn[i]; 
                    if (qn[i] > q_max[i]) q_max[i] = qn[i]; 
                }
            }
            PetscScalar *d_min, *d_max;
            PetscCall(DMPlexPointLocalRef(dmQ, c, min_ptr, &d_min)); 
            PetscCall(DMPlexPointLocalRef(dmQ, c, max_ptr, &d_max));
            for(int i=0; i<Model<T>::n_dof_q; ++i) { d_min[i] = q_min[i]; d_max[i] = q_max[i]; }
        }
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr)); 
        PetscCall(VecRestoreArray(V_min_loc, &min_ptr)); PetscCall(VecRestoreArray(V_max_loc, &max_ptr));
        return PETSC_SUCCESS;
    }

    // Field-level, mesh-aware refresh of the spatial-derivative aux (REQ-65):
    // de-interleave the local Q/Qaux into per-field arrays, call the model's
    // field-level update_aux_variables (which fills the derivative slots via the
    // real mesh-aware compute_derivative), then scatter back. No-op for models
    // without the field-level overload (e.g. plain SWE/SME with only local aux).
    PetscErrorCode RefreshDerivativeAux(Vec X_loc, Vec A_loc) {
        if constexpr (HasFieldAuxUpdate<T>::value) {
            const int nq = Model<T>::n_dof_q, na = Model<T>::n_dof_qaux;
            PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
            PetscInt ncell = cEnd - cStart;
            PetscInt size_q, size_a;
            PetscCall(VecGetLocalSize(X_loc, &size_q)); PetscCall(VecGetLocalSize(A_loc, &size_a));
            std::vector<std::vector<T>> Qf(nq, std::vector<T>(ncell, (T)0));
            std::vector<std::vector<T>> Af(na, std::vector<T>(ncell, (T)0));
            PetscScalar *x_ptr, *a_ptr;
            PetscCall(VecGetArray(X_loc, &x_ptr)); PetscCall(VecGetArray(A_loc, &a_ptr));
            PetscSection sQ, sA; PetscCall(DMGetLocalSection(dmQ, &sQ)); PetscCall(DMGetLocalSection(dmAux, &sA));
            for (PetscInt c = cStart; c < cEnd; ++c) {
                PetscInt oq, oa; PetscCall(PetscSectionGetOffset(sQ, c, &oq)); PetscCall(PetscSectionGetOffset(sA, c, &oa));
                if (oq < 0 || (oq + nq) > size_q || oa < 0 || (oa + na) > size_a) continue;
                for (int i = 0; i < nq; ++i) Qf[i][c - cStart] = x_ptr[oq + i];
                for (int k = 0; k < na; ++k) Af[k][c - cStart] = a_ptr[oa + k];
            }
            std::vector<T*> Qp(nq), Ap(na);
            for (int i = 0; i < nq; ++i) Qp[i] = Qf[i].data();
            for (int k = 0; k < na; ++k) Ap[k] = Af[k].data();
            ZoomyMesh mesh{dmQ, cStart, cEnd};
            Model<T>::update_aux_variables(Qp.data(), Ap.data(), (T)0, mesh);
            for (PetscInt c = cStart; c < cEnd; ++c) {
                PetscInt oa; PetscCall(PetscSectionGetOffset(sA, c, &oa));
                if (oa < 0 || (oa + na) > size_a) continue;
                for (int k = 0; k < na; ++k) a_ptr[oa + k] = Af[k][c - cStart];
            }
            PetscCall(VecRestoreArray(X_loc, &x_ptr)); PetscCall(VecRestoreArray(A_loc, &a_ptr));
        }
        return PETSC_SUCCESS;
    }

    void SetRefreshDerivativeAux(bool b) { do_refresh_deriv_aux = b; }

    PetscErrorCode UpdateState(Vec X_loc, Vec A_loc) {
        // Fill the spatial-derivative aux first (field-level/mesh-aware); the
        // per-cell pass below then preserves them and computes the local aux.
        // Skippable when no kernel consumes the derivative-aux (e.g. SWE/SME(0)
        // Rusanov use only the local hinv) — the per-step Green-Gauss is then
        // pure overhead.
        if (do_refresh_deriv_aux) PetscCall(RefreshDerivativeAux(X_loc, A_loc));
        PetscScalar *x_ptr, *a_ptr;
        PetscCall(VecGetArray(X_loc, &x_ptr)); PetscCall(VecGetArray(A_loc, &a_ptr));
        PetscInt size_q, size_a;
        PetscCall(VecGetLocalSize(X_loc, &size_q)); PetscCall(VecGetLocalSize(A_loc, &size_a));
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        PetscSection sQ, sAux; PetscCall(DMGetLocalSection(dmQ, &sQ)); PetscCall(DMGetLocalSection(dmAux, &sAux));
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
            PetscInt offQ, offA; PetscCall(PetscSectionGetOffset(sQ, c, &offQ)); PetscCall(PetscSectionGetOffset(sAux, c, &offA));
            if (offQ >= 0 && (offQ + Model<T>::n_dof_q) <= size_q && offA >= 0 && (offA + Model<T>::n_dof_qaux) <= size_a) {
                PetscScalar *q = &x_ptr[offQ]; PetscScalar *a = &a_ptr[offA];
                // Q post-update (the wet/dry momentum clamp): caps |u| and zeros
                // momentum below wet_dry_eps. This is the positivity safety net
                // the numpy/jax solver applies via update_variables every step —
                // without it momentum runs away at the advancing front and h goes
                // negative. Applied only if the model emits it (SWE/MalpassetSME).
                if constexpr (HasUpdateVariables<T>::value) {
                    auto res_q = Model<T>::update_variables(q, a, parameters.data(), 0.0);
                    for(int i=0; i<Model<T>::n_dof_q; ++i) q[i] = res_q[i];
                }
                auto res_a = Model<T>::update_aux_variables(q, a, parameters.data(), 0.0);
                for(int i=0; i<Model<T>::n_dof_qaux; ++i) a[i] = res_a[i];
            }
        }
        PetscCall(VecRestoreArray(X_loc, &x_ptr)); PetscCall(VecRestoreArray(A_loc, &a_ptr));
        return PETSC_SUCCESS;
    }

    void SetWBPositivity(bool b, T eps) { wb_positivity = b; wet_dry_eps = eps; }
    void SetMoodSkipTheta(bool b) { mood_skip_theta = b; }

    // Per-cell order-2 limiting pass (eta-WB + Zhang-Shu): transform the raw
    // gradient G (grad of Q=[b,h,mom...]) into EFFECTIVE W-gradients in place
    // (slot b -> phi_b*grad b; slot h -> phi_b*grad b + theta*sh; mom ->
    // theta*phi_mom*grad mom). Owned cells only (G is global); ghosts get the
    // limited values via the subsequent global->local scatter. SWE/SME(0)
    // layout: b=0, h=1, momentum = the remaining rows.
    PetscErrorCode LimitGradientsWB(Vec X_loc, Vec G_global) {
        const int nq = Model<T>::n_dof_q, dim = Model<T>::dimension;
        const int B = 0, H = 1;
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));
        if (!topology_setup) PetscCall(SetupTopology());
        Vec cellGeom, faceGeom; PetscCall(DMPlexGetGeometryFVM(dmQ, &faceGeom, &cellGeom, NULL));
        const PetscScalar *cG, *fG; PetscCall(VecGetArrayRead(cellGeom, &cG)); PetscCall(VecGetArrayRead(faceGeom, &fG));
        DM dmCell; PetscCall(VecGetDM(cellGeom, &dmCell)); PetscSection secCell; PetscCall(DMGetLocalSection(dmCell, &secCell));
        DM dmFace; PetscCall(VecGetDM(faceGeom, &dmFace)); PetscSection secFace; PetscCall(DMGetLocalSection(dmFace, &secFace));
        const PetscScalar *x_ptr; PetscCall(VecGetArrayRead(X_loc, &x_ptr));
        PetscScalar *g_ptr; PetscCall(VecGetArray(G_global, &g_ptr));

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscScalar *grad_c;
            PetscCall(DMPlexPointGlobalRef(dmGrad, c, g_ptr, &grad_c));
            if (!grad_c) continue;                       // not owned
            const PetscScalar *qc; PetscCall(DMPlexPointLocalRead(dmQ, c, x_ptr, &qc));
            PetscInt co; PetscCall(PetscSectionGetOffset(secCell, c, &co));
            const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cG[co];

            // raw gradients (copy out before overwriting)
            std::vector<T> g(nq*dim);
            for (int k = 0; k < nq*dim; ++k) g[k] = grad_c[k];
            std::vector<T> geta(dim);
            for (int d = 0; d < dim; ++d) geta[d] = g[H*dim+d] + g[B*dim+d];   // grad eta = grad h + grad b

            T h_bar = qc[H], b_bar = qc[B];
            // W neighbour extrema (W = b, eta, mom...) including self
            std::vector<T> Wc(nq), Wmin(nq), Wmax(nq);
            for (int i = 0; i < nq; ++i) Wc[i] = (i==H) ? (b_bar+h_bar) : qc[i];
            for (int i = 0; i < nq; ++i) { Wmin[i] = Wc[i]; Wmax[i] = Wc[i]; }
            int is = neigh_offsets[c-cStart], ie = neigh_offsets[c-cStart+1];
            for (int k = is; k < ie; ++k) {
                const PetscScalar *qn; PetscCall(DMPlexPointLocalRead(dmQ, neigh_list[k], x_ptr, &qn));
                for (int i = 0; i < nq; ++i) { T w = (i==H) ? (qn[B]+qn[H]) : qn[i];
                    if (w < Wmin[i]) Wmin[i] = w; if (w > Wmax[i]) Wmax[i] = w; }
            }
            // Venkatakrishnan eps^2 = vol^(2/dim)
            T vol = cg->volume; T eps2 = std::pow(vol, 2.0/dim);
            // per-cell phi = min over faces of the Venkat function (per W var)
            std::vector<T> phi(nq, 1.0);
            const PetscInt *cone; PetscInt ncone; PetscCall(DMPlexGetConeSize(dmQ, c, &ncone)); PetscCall(DMPlexGetCone(dmQ, c, &cone));
            auto gW = [&](int i, int d){ return (i==H) ? geta[d] : g[i*dim+d]; };
            for (int fi = 0; fi < ncone; ++fi) {
                PetscInt f = cone[fi]; PetscInt fo; PetscCall(PetscSectionGetOffset(secFace, f, &fo));
                const PetscFVFaceGeom *fgeo = (const PetscFVFaceGeom*)&fG[fo];
                T r[3]; for (int d=0; d<dim; ++d) r[d] = fgeo->centroid[d] - cg->centroid[d];
                for (int i = 0; i < nq; ++i) {
                    T delta = 0; for (int d=0; d<dim; ++d) delta += gW(i,d)*r[d];
                    if (std::abs(delta) < 1e-14) continue;
                    T dm = (delta > 0) ? (Wmax[i]-Wc[i]) : (Wmin[i]-Wc[i]);
                    T num = dm*dm + eps2 + 2*delta*dm, den = dm*dm + 2*delta*delta + delta*dm + eps2;
                    T pf = (den > 1e-30) ? num/den : 1.0; pf = std::max((T)0, std::min((T)1, pf));
                    if (pf < phi[i]) phi[i] = pf;
                }
            }
            if (h_bar < wet_dry_eps) for (int i=0;i<nq;++i) phi[i] = 0.0;   // dry -> O1
            // tie b and momentum slopes to eta
            phi[B] = std::min(phi[B], phi[H]);
            for (int i = 0; i < nq; ++i) if (i!=B && i!=H) phi[i] = std::min(phi[i], phi[H]);
            // Zhang-Shu theta on the conservative depth deviation sh = phi_eta*grad eta - phi_b*grad b
            std::vector<T> sh(dim); for (int d=0; d<dim; ++d) sh[d] = phi[H]*geta[d] - phi[B]*g[B*dim+d];
            T hmin = h_bar;
            for (int fi = 0; fi < ncone; ++fi) {
                PetscInt f = cone[fi]; PetscInt fo; PetscCall(PetscSectionGetOffset(secFace, f, &fo));
                const PetscFVFaceGeom *fgeo = (const PetscFVFaceGeom*)&fG[fo];
                T hf = h_bar; for (int d=0; d<dim; ++d) hf += sh[d]*(fgeo->centroid[d]-cg->centroid[d]);
                if (hf < hmin) hmin = hf;
            }
            T theta = 1.0;
            if (!mood_skip_theta && hmin < 0.0) theta = std::max((T)0, std::min((T)1, h_bar/std::max(h_bar - hmin, (T)1e-14)));
            // write effective W-gradients back
            for (int d = 0; d < dim; ++d) {
                grad_c[B*dim+d] = phi[B]*g[B*dim+d];                       // grad b_eff
                grad_c[H*dim+d] = phi[B]*g[B*dim+d] + theta*sh[d];         // grad eta_eff
            }
            for (int i = 0; i < nq; ++i) if (i!=B && i!=H)
                for (int d = 0; d < dim; ++d) grad_c[i*dim+d] = theta*phi[i]*g[i*dim+d];
        }
        PetscCall(VecRestoreArrayRead(cellGeom, &cG)); PetscCall(VecRestoreArrayRead(faceGeom, &fG));
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr)); PetscCall(VecRestoreArray(G_global, &g_ptr));
        return PETSC_SUCCESS;
    }

    PetscErrorCode FormRHS(PetscReal time, Vec X_global, Vec F_global) {
        PetscCall(VecZeroEntries(F_global));
        Vec X_loc, A_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc)); 
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc)); 
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc)); 
        
        PetscCall(UpdateState(X_loc, A_loc)); 
        if (gradient) PetscCall(UpdateNeighborBounds(X_loc, A_loc));

        Vec G_global = NULL; Vec G_loc = NULL; PetscScalar *g_ptr = NULL;
        if (gradient) {
             PetscCall(DMCreateGlobalVector(dmGrad, &G_global));
             PetscCall(gradient->Compute(dmQ, X_global, dmGrad, G_global, boundary_map));
             if (wb_positivity) PetscCall(LimitGradientsWB(X_loc, G_global));
             PetscCall(DMGetLocalVector(dmGrad, &G_loc));
             PetscCall(DMGlobalToLocalBegin(dmGrad, G_global, INSERT_VALUES, G_loc));
             PetscCall(DMGlobalToLocalEnd(dmGrad, G_global, INSERT_VALUES, G_loc));
             PetscCall(VecGetArray(G_loc, &g_ptr));
        }

        Vec F_loc; PetscCall(DMGetLocalVector(dmQ, &F_loc)); PetscCall(VecZeroEntries(F_loc));
        PetscCall(ComputeFluxes(time, X_loc, A_loc, g_ptr, F_loc));
        PetscCall(DMLocalToGlobalBegin(dmQ, F_loc, ADD_VALUES, F_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, F_loc, ADD_VALUES, F_global));

        if (G_loc) { PetscCall(VecRestoreArray(G_loc, &g_ptr)); PetscCall(DMRestoreLocalVector(dmGrad, &G_loc)); }
        if (G_global) PetscCall(VecDestroy(&G_global));

        PetscCall(DMRestoreLocalVector(dmQ, &X_loc)); PetscCall(DMRestoreLocalVector(dmAux, &A_loc)); PetscCall(DMRestoreLocalVector(dmQ, &F_loc));
        return PETSC_SUCCESS;
    }

    // Local MOOD: O1 (PCM, no gradients) RHS over X_global, restricted to faces
    // adjacent to troubled cells. F_global[c] is the first-order forward-Euler
    // dQ/dt for each troubled cell c (only troubled cells are meaningful). Cheap:
    // only troubled-adjacent faces are visited.
    PetscErrorCode FormRHSTroubledO1(PetscReal time, Vec X_global, Vec F_global, const std::vector<uint8_t>& mask) {
        PetscCall(VecZeroEntries(F_global));
        Vec X_loc, A_loc, F_loc;
        PetscCall(DMGetLocalVector(dmQ, &X_loc));
        PetscCall(DMGlobalToLocalBegin(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGlobalToLocalEnd(dmQ, X_global, INSERT_VALUES, X_loc));
        PetscCall(DMGetLocalVector(dmAux, &A_loc));
        PetscCall(UpdateState(X_loc, A_loc));
        PetscCall(DMGetLocalVector(dmQ, &F_loc)); PetscCall(VecZeroEntries(F_loc));
        troubled_filter = &mask;
        PetscCall(ComputeFluxes(time, X_loc, A_loc, NULL /*PCM -> first order*/, F_loc));
        troubled_filter = nullptr;
        PetscCall(DMLocalToGlobalBegin(dmQ, F_loc, ADD_VALUES, F_global));
        PetscCall(DMLocalToGlobalEnd(dmQ, F_loc, ADD_VALUES, F_global));
        PetscCall(DMRestoreLocalVector(dmQ, &X_loc));
        PetscCall(DMRestoreLocalVector(dmAux, &A_loc));
        PetscCall(DMRestoreLocalVector(dmQ, &F_loc));
        return PETSC_SUCCESS;
    }

    PetscErrorCode ComputeFluxes(PetscReal time, Vec X_loc, Vec A_loc, const PetscScalar* g_ptr, Vec F_loc) {
        const PetscScalar *x_ptr, *a_ptr; PetscScalar *f_ptr;
        PetscCall(VecGetArrayRead(X_loc, &x_ptr)); PetscCall(VecGetArrayRead(A_loc, &a_ptr)); PetscCall(VecGetArray(F_loc, &f_ptr));
        PetscInt size_f; PetscCall(VecGetLocalSize(F_loc, &size_f));

        Vec cellGeom, faceGeom; PetscCall(DMPlexGetGeometryFVM(dmQ, &faceGeom, &cellGeom, NULL));
        const PetscScalar *fGeom_ptr, *cGeom_ptr;
        PetscCall(VecGetArrayRead(faceGeom, &fGeom_ptr)); PetscCall(VecGetArrayRead(cellGeom, &cGeom_ptr));
        DM dmFace; PetscCall(VecGetDM(faceGeom, &dmFace)); PetscSection secFace; PetscCall(DMGetLocalSection(dmFace, &secFace));
        DM dmCell; PetscCall(VecGetDM(cellGeom, &dmCell)); PetscSection secCell; PetscCall(DMGetLocalSection(dmCell, &secCell));
        PetscSection sQ; PetscCall(DMGetLocalSection(dmQ, &sQ));

        const PetscScalar *min_ptr = NULL, *max_ptr = NULL;
        if (V_min_loc) { PetscCall(VecGetArrayRead(V_min_loc, &min_ptr)); PetscCall(VecGetArrayRead(V_max_loc, &max_ptr)); }

        PetscInt fStart, fEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 1, &fStart, &fEnd));
        DMLabel label; PetscCall(DMGetLabel(dmQ, "Face Sets", &label));
        PetscInt dim = Model<T>::dimension;
        PetscInt cStart, cEnd; PetscCall(DMPlexGetHeightStratum(dmQ, 0, &cStart, &cEnd));

        for (PetscInt f = fStart; f < fEnd; ++f) {
            if (!IsOwned(dmQ, f)) continue;
            PetscInt off; PetscCall(PetscSectionGetOffset(secFace, f, &off)); const PetscFVFaceGeom *fg = (const PetscFVFaceGeom*)&fGeom_ptr[off];
            PetscScalar n_hat[3] = {0}; PetscReal area = 0; for(int d=0; d<dim; ++d) area += fg->normal[d]*fg->normal[d]; area = std::sqrt(area);
            if(area <= 1e-15) continue;
            for(int d=0; d<dim; ++d) n_hat[d] = fg->normal[d] / area;

            const PetscInt *cells; PetscInt num_cells; PetscCall(DMPlexGetSupportSize(dmQ, f, &num_cells)); PetscCall(DMPlexGetSupport(dmQ, f, &cells));

            // Local-MOOD: skip faces not touching any troubled cell.
            if (troubled_filter) {
                bool touch = false;
                for (PetscInt s = 0; s < num_cells; ++s)
                    if (cells[s] >= cStart && cells[s] < cEnd && (*troubled_filter)[cells[s] - cStart] == 1) { touch = true; break; }
                if (!touch) continue;
            }

            if (num_cells == 2) {
                const PetscScalar *qL_cell, *qR_cell; const PetscScalar *aL_cell, *aR_cell;
                const PetscScalar *gL_cell = NULL, *gR_cell = NULL;

                PetscCall(DMPlexPointLocalRead(dmQ, cells[0], x_ptr, &qL_cell)); PetscCall(DMPlexPointLocalRead(dmQ, cells[1], x_ptr, &qR_cell));
                PetscCall(DMPlexPointLocalRead(dmAux, cells[0], a_ptr, &aL_cell)); PetscCall(DMPlexPointLocalRead(dmAux, cells[1], a_ptr, &aR_cell));

                if (g_ptr) { PetscCall(DMPlexPointLocalRead(dmGrad, cells[0], g_ptr, &gL_cell)); PetscCall(DMPlexPointLocalRead(dmGrad, cells[1], g_ptr, &gR_cell)); }

                if (!cell_orders.empty()) {
                    if (cells[0] >= cStart && cells[0] < cEnd && cell_orders[cells[0] - cStart] == 1) { gL_cell = nullptr; }
                    if (cells[1] >= cStart && cells[1] < cEnd && cell_orders[cells[1] - cStart] == 1) { gR_cell = nullptr; }
                }

                PetscInt offL, offR; PetscCall(PetscSectionGetOffset(secCell, cells[0], &offL)); PetscCall(PetscSectionGetOffset(secCell, cells[1], &offR));
                const PetscFVCellGeom *cgL = (const PetscFVCellGeom*)&cGeom_ptr[offL]; const PetscFVCellGeom *cgR = (const PetscFVCellGeom*)&cGeom_ptr[offR];
                
                PetscScalar qL_face[Model<T>::n_dof_q], qR_face[Model<T>::n_dof_q];
                const PetscScalar *minL=NULL, *maxL=NULL, *minR=NULL, *maxR=NULL;
                if (min_ptr) { 
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[0], min_ptr, &minL)); PetscCall(DMPlexPointLocalRead(dmQ, cells[0], max_ptr, &maxL)); 
                    PetscCall(DMPlexPointLocalRead(dmQ, cells[1], min_ptr, &minR)); PetscCall(DMPlexPointLocalRead(dmQ, cells[1], max_ptr, &maxR)); 
                }
                reconstructor->Reconstruct(qL_cell, gL_cell, cgL->centroid, fg->centroid, minL, maxL, qL_face);
                reconstructor->Reconstruct(qR_cell, gR_cell, cgR->centroid, fg->centroid, minR, maxR, qR_face);
                
                PetscScalar aL_face[Model<T>::n_dof_qaux], aR_face[Model<T>::n_dof_qaux];
                reconstructorAux->Reconstruct(aL_cell, nullptr, cgL->centroid, fg->centroid, NULL, NULL, aL_face);
                reconstructorAux->Reconstruct(aR_cell, nullptr, cgR->centroid, fg->centroid, NULL, NULL, aR_face);

                auto res_aL = Model<T>::update_aux_variables(qL_face, aL_face, parameters.data(), 0.0); for(int i=0; i<Model<T>::n_dof_qaux; ++i) aL_face[i] = res_aL[i];
                auto res_aR = Model<T>::update_aux_variables(qR_face, aR_face, parameters.data(), 0.0); for(int i=0; i<Model<T>::n_dof_qaux; ++i) aR_face[i] = res_aR[i];

                SimpleArray<T, Model<T>::n_dof_q> flux; for(int i=0; i<Model<T>::n_dof_q; ++i) flux[i] = 0.0;
                if (cons_flux_kernel) flux = cons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);

                PetscInt offFL, offFR; PetscCall(PetscSectionGetOffset(sQ, cells[0], &offFL)); PetscCall(PetscSectionGetOffset(sQ, cells[1], &offFR));
                PetscScalar *fL = (offFL >= 0 && offFL + Model<T>::n_dof_q <= size_f) ? &f_ptr[offFL] : nullptr;
                PetscScalar *fR = (offFR >= 0 && offFR + Model<T>::n_dof_q <= size_f) ? &f_ptr[offFR] : nullptr;

                if (noncons_flux_kernel) {
                    auto nc_stacked = noncons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);
                    auto* nc_into_right = nc_stacked.data; auto* nc_into_left = nc_stacked.data + Model<T>::n_dof_q;
                    for(int i=0; i<Model<T>::n_dof_q; ++i) { if (fL) fL[i] -= nc_into_left[i] * area; if (fR) fR[i] -= nc_into_right[i] * area; }
                }
                for(int i=0; i<Model<T>::n_dof_q; ++i) { if (fL) fL[i] -= flux[i] * area; if (fR) fR[i] += flux[i] * area; }

                // --- Diffusive flux (if model defines diffusive_flux) ---
                if constexpr (HasDiffusiveFlux<T>::value) {
                    if (g_ptr && gL_cell && gR_cell) {
                        // Face-averaged gradient: 0.5 * (gradQ_L + gradQ_R)
                        constexpr int n_gradQ = Model<T>::n_dof_q * Model<T>::dimension;
                        T gradQ_face[n_gradQ];
                        for (int k = 0; k < n_gradQ; ++k) {
                            gradQ_face[k] = 0.5 * (gL_cell[k] + gR_cell[k]);
                        }
                        // Face-averaged state and aux for the diffusive flux evaluation
                        T q_face[Model<T>::n_dof_q], a_face[Model<T>::n_dof_qaux];
                        for (int i = 0; i < Model<T>::n_dof_q; ++i)
                            q_face[i] = 0.5 * (qL_face[i] + qR_face[i]);
                        for (int i = 0; i < Model<T>::n_dof_qaux; ++i)
                            a_face[i] = 0.5 * (aL_face[i] + aR_face[i]);

                        auto diff_flux = Model<T>::diffusive_flux(q_face, a_face, gradQ_face, parameters.data());

                        // Diffusive flux dotted with face normal (already scaled by area via n_hat * area)
                        // Convention: diffusive flux is ADDED to residual (it opposes convective flux sign)
                        for (int i = 0; i < Model<T>::n_dof_q; ++i) {
                            if (fL) fL[i] += diff_flux[i] * area;
                            if (fR) fR[i] -= diff_flux[i] * area;
                        }
                    }
                }

            } else if (num_cells == 1) {
                // Boundary Face
                PetscInt tag_id; PetscCall(DMLabelGetValue(label, f, &tag_id));
                if (boundary_map.count(tag_id)) {
                    PetscInt bc_idx = boundary_map.at(tag_id);
                    PetscInt cL = cells[0];
                    const PetscScalar *qL_cell; PetscCall(DMPlexPointLocalRead(dmQ, cL, x_ptr, &qL_cell));
                    const PetscScalar *aL_cell; PetscCall(DMPlexPointLocalRead(dmAux, cL, a_ptr, &aL_cell));
                    const PetscScalar *gL_cell = NULL;
                    if (g_ptr) { PetscCall(DMPlexPointLocalRead(dmGrad, cL, g_ptr, &gL_cell)); }
                    if (!cell_orders.empty()) { if (cL >= cStart && cL < cEnd && cell_orders[cL - cStart] == 1) { gL_cell = nullptr; } }
                    PetscInt offL; PetscCall(PetscSectionGetOffset(secCell, cL, &offL));
                    const PetscFVCellGeom *cgL = (const PetscFVCellGeom*)&cGeom_ptr[offL];
                    PetscScalar qL_face[Model<T>::n_dof_q];
                    const PetscScalar *minL=NULL, *maxL=NULL;
                    if (min_ptr) { PetscCall(DMPlexPointLocalRead(dmQ, cL, min_ptr, &minL)); PetscCall(DMPlexPointLocalRead(dmQ, cL, max_ptr, &maxL)); }
                    reconstructor->Reconstruct(qL_cell, gL_cell, cgL->centroid, fg->centroid, minL, maxL, qL_face);
                    PetscScalar aL_face[Model<T>::n_dof_qaux];
                    reconstructorAux->Reconstruct(aL_cell, nullptr, cgL->centroid, fg->centroid, NULL, NULL, aL_face);
                    auto res_aL = Model<T>::update_aux_variables(qL_face, aL_face, parameters.data(), 0.0);
                    for(int i=0; i<Model<T>::n_dof_qaux; ++i) aL_face[i] = res_aL[i];
                    auto qR_arr = Model<T>::boundary_conditions(bc_idx, qL_face, aL_face, parameters.data(), n_hat, fg->centroid, time, 0.0);
                    PetscScalar *qR_face = qR_arr.data;
                    PetscScalar aR_face[Model<T>::n_dof_qaux];
                    auto res_aR = Model<T>::update_aux_variables(qR_face, aL_face, parameters.data(), 0.0);
                    for(int i=0; i<Model<T>::n_dof_qaux; ++i) aR_face[i] = res_aR[i];
                    SimpleArray<T, Model<T>::n_dof_q> flux; for(int i=0; i<Model<T>::n_dof_q; ++i) flux[i] = 0.0;
                    if (cons_flux_kernel) flux = cons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);
                    PetscInt offFL; PetscCall(PetscSectionGetOffset(sQ, cL, &offFL));
                    PetscScalar *fL = (offFL >= 0 && offFL + Model<T>::n_dof_q <= size_f) ? &f_ptr[offFL] : nullptr;
                    if (noncons_flux_kernel) {
                        auto nc_stacked = noncons_flux_kernel(qL_face, qR_face, aL_face, aR_face, parameters.data(), n_hat);
                        auto* nc_into_left  = nc_stacked.data + Model<T>::n_dof_q;
                        if (fL) { for(int i=0; i<Model<T>::n_dof_q; ++i) fL[i] -= nc_into_left[i] * area; }
                    }
                    if (fL) { for(int i=0; i<Model<T>::n_dof_q; ++i) fL[i] -= flux[i] * area; }

                    // --- Diffusive flux at boundary (if model defines diffusive_flux) ---
                    if constexpr (HasDiffusiveFlux<T>::value) {
                        if (g_ptr && gL_cell) {
                            // At boundaries, use the interior cell gradient (one-sided)
                            const T* gradQ_face = gL_cell;

                            T q_face[Model<T>::n_dof_q], a_face[Model<T>::n_dof_qaux];
                            for (int i = 0; i < Model<T>::n_dof_q; ++i)
                                q_face[i] = 0.5 * (qL_face[i] + qR_face[i]);
                            for (int i = 0; i < Model<T>::n_dof_qaux; ++i)
                                a_face[i] = aL_face[i];

                            auto diff_flux = Model<T>::diffusive_flux(q_face, a_face, gradQ_face, parameters.data());
                            if (fL) {
                                for (int i = 0; i < Model<T>::n_dof_q; ++i) fL[i] += diff_flux[i] * area;
                            }
                        }
                    }
                }
            }
        }
        
        for(PetscInt c=cStart; c<cEnd; ++c) {
             PetscInt off; PetscCall(PetscSectionGetOffset(secCell, c, &off)); const PetscFVCellGeom *cg = (const PetscFVCellGeom*)&cGeom_ptr[off];
             PetscInt offF; PetscCall(PetscSectionGetOffset(sQ, c, &offF));
             PetscScalar *f_cell = (offF >= 0 && offF + Model<T>::n_dof_q <= size_f) ? &f_ptr[offF] : nullptr;
             if (f_cell && cg->volume > 1e-15) { for(int i=0; i<Model<T>::n_dof_q; ++i) f_cell[i] /= cg->volume; }
        }
        if (V_min_loc) { PetscCall(VecRestoreArrayRead(V_min_loc, &min_ptr)); PetscCall(VecRestoreArrayRead(V_max_loc, &max_ptr)); }
        PetscCall(VecRestoreArrayRead(X_loc, &x_ptr)); PetscCall(VecRestoreArrayRead(A_loc, &a_ptr)); PetscCall(VecRestoreArray(F_loc, &f_ptr));
        PetscCall(VecRestoreArrayRead(faceGeom, &fGeom_ptr)); PetscCall(VecRestoreArrayRead(cellGeom, &cGeom_ptr));
        return PETSC_SUCCESS;
    }
};
#endif