#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <map> 
#include <vector>
#include "nlohmann/json.hpp" 

using json = nlohmann::json;

struct SolverSettings {
    double t_end = 1.0;
    double cfl = 0.5;
    bool use_deep_adjacency = true;
    int reconstruction_order = 1;
    double min_dt = 1.0e-12;
    std::string limiter = "venkatakrishnan";  // "venkatakrishnan" (default), "tvd", or "none"
    // Refresh the mesh-derivative aux (compute_derivative) every step. Needed
    // only when a kernel consumes the derivative-aux (diffusion / higher SME);
    // for SWE/SME(0) Rusanov (only hinv used) set false to skip the per-step
    // Green-Gauss overhead.
    bool refresh_derivative_aux = true;
    // Order-2 positivity: "zhang_shu" = eta-WB reconstruction + Zhang-Shu theta
    // cap (provably h>=0; auto-clamps CFL<=1/6), "none" = plain MUSCL limiter.
    std::string positivity = "none";
    double wet_dry_eps = 1.0e-2;
    // Time integration, jax-style: pick the integrator from settings instead of
    // hard-coding it. "splitting" = explicit transport (SSP-RK) + implicit source
    // (default, like jax's explicit HyperbolicSolver); "imex" = explicit transport
    // + implicit source via TSARKIMEX (like jax's IMEXSourceSolverJax, for stiff
    // sources); "implicit" = fully implicit BDF2.
    std::string time_integration = "splitting";  // "splitting" | "imex" | "implicit"
    std::string method = "muscl";                 // "muscl" (FV) | "chorin" (VAM pressure-split)
    // Order-2 explicit integrator = s-stage SSP-RK2 (2nd order, SSP coeff s-1).
    // 2 = Heun (cheapest, weakest positivity margin); 5 = SSP coeff 4 (default,
    // roundoff positivity with MOOD at ~half the cost of 4th-order SSP-RK104).
    int ssp_stages = 5;
};

struct IOSettings {
    std::string directory = "output";
    std::string filename = "sol";
    int snapshots = 10;
    std::string snapshot_logic = "snap"; 
    bool clean_directory = false;
    std::string mesh_path = "";
    std::string mesh_label = "Face Sets"; 
    bool restart = false;
    std::string restart_file = "";
    
    // --- Initial Condition Config ---
    std::string initial_condition_file = "";
    std::vector<int> initial_condition_mask; // If empty, overwrite all

    // --- 3D Output Settings ---
    bool write_3d = false;
    int n_layers_3d = 10;
    std::string output_3d_name = "sol_3d";
};

struct ModelSettings {
    std::map<std::string, double> parameters; 
};

struct Settings {
    std::string name;
    IOSettings io;
    SolverSettings solver;
    ModelSettings model;

    static Settings from_json(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Error: Could not open settings file: " << path << std::endl;
            exit(1);
        }
        json j;
        f >> j;

        Settings s;
        s.name = j.value("name", "Simulation");

        if (j.contains("io")) {
            auto& jio = j["io"];
            s.io.directory = jio.value("directory", "output");
            s.io.filename = jio.value("filename", "sol");
            s.io.snapshots = jio.value("snapshots", 10);
            s.io.snapshot_logic = jio.value("snapshot_logic", "snap");
            s.io.clean_directory = jio.value("clean_directory", false);
            s.io.mesh_path = jio.value("mesh_path", "");
            s.io.mesh_label = jio.value("mesh_label", "Face Sets"); 
            
            s.io.initial_condition_file = jio.value("initial_condition_file", "");
            
            // Parse Mask array: [0, 1, 2, 4]
            if (jio.contains("initial_condition_mask")) {
                for (auto& element : jio["initial_condition_mask"]) {
                    if (element.is_number_integer()) {
                        s.io.initial_condition_mask.push_back(element.get<int>());
                    }
                }
            }

            s.io.write_3d = jio.value("write_3d", false);
            s.io.n_layers_3d = jio.value("n_layers_3d", 10);
            s.io.output_3d_name = jio.value("output_3d_name", "sol_3d");
        }

        if (j.contains("solver")) {
            auto& jsol = j["solver"];
            s.solver.t_end = jsol.value("t_end", 1.0);
            s.solver.cfl = jsol.value("cfl", 0.5);
            s.solver.use_deep_adjacency = jsol.value("use_deep_adjacency", true); 
            s.solver.reconstruction_order = jsol.value("reconstruction_order", 1); 
            s.solver.min_dt = jsol.value("min_dt", 1.0e-12);
            s.solver.limiter = jsol.value("limiter", "venkatakrishnan");
            s.solver.time_integration = jsol.value("time_integration", "splitting");
            s.solver.method = jsol.value("method", "muscl");
            s.solver.refresh_derivative_aux = jsol.value("refresh_derivative_aux", true);
            s.solver.positivity = jsol.value("positivity", "none");
            s.solver.wet_dry_eps = jsol.value("wet_dry_eps", 1.0e-2);
            s.solver.ssp_stages = jsol.value("ssp_stages", 5);
        }

        if (j.contains("model")) {
            if (j["model"].contains("parameters")) {
                for (auto& [key, val] : j["model"]["parameters"].items()) {
                    if (val.is_number()) {
                        s.model.parameters[key] = val.get<double>();
                    }
                }
            }
        }

        return s;
    }
};

#endif