# AGENT.md - Solvers Module

## Mission
Provide comprehensive linear and nonlinear solution algorithms for finite element systems, supporting direct and iterative methods, eigenvalue problems, time integration, and optimization, with seamless integration of external solver libraries and GPU acceleration.

## Architecture Philosophy
- **Algorithm-Rich**: Multiple solver strategies for different problem types
- **Preconditioner-Aware**: Deep integration with preconditioning strategies
- **Scale-Adaptive**: From small dense to massive sparse systems
- **Hardware-Flexible**: CPU, GPU, and distributed implementations
- **Library-Integrated**: Leverage PETSc, Trilinos, and other proven libraries

## Directory Structure

```
solvers/
├── README.md                        # Module overview
├── AGENT.md                         # This document
├── CMakeLists.txt                   # Build configuration
│
├── base/                            # Common solver infrastructure
│   ├── solver_base.hpp             # Base solver interface
│   ├── linear_solver.hpp           # Linear solver base
│   ├── nonlinear_solver.hpp        # Nonlinear solver base
│   ├── solver_traits.hpp           # Solver type traits
│   ├── convergence_criteria.hpp    # Convergence tests
│   ├── solver_monitor.hpp          # Solution monitoring
│   ├── solver_statistics.hpp       # Performance statistics
│   └── solver_factory.hpp          # Factory for solver creation
│
├── direct/                          # Direct solvers
│   ├── direct_solver_base.hpp      # Direct solver interface
│   ├── dense/
│   │   ├── lu_solver.hpp           # LU decomposition
│   │   ├── qr_solver.hpp           # QR decomposition  
│   │   ├── svd_solver.hpp          # SVD solver
│   │   ├── cholesky_solver.hpp     # Cholesky for SPD
│   │   └── ldlt_solver.hpp         # LDLT for symmetric
│   ├── sparse/
│   │   ├── sparse_lu.hpp           # Sparse LU
│   │   ├── sparse_cholesky.hpp     # Sparse Cholesky
│   │   ├── sparse_qr.hpp           # Sparse QR
│   │   ├── multifrontal.hpp        # Multifrontal methods
│   │   └── supernodal.hpp          # Supernodal factorization
│   ├── banded/
│   │   ├── band_solver.hpp         # Banded matrix solver
│   │   ├── tridiagonal.hpp         # Thomas algorithm
│   │   └── block_tridiagonal.hpp   # Block tridiagonal
│   ├── structured/
│   │   ├── fft_solver.hpp          # FFT-based (periodic)
│   │   ├── cyclic_reduction.hpp    # Cyclic reduction
│   │   └── fast_poisson.hpp        # Fast Poisson solver
│   └── parallel/
│       ├── distributed_lu.hpp      # Parallel LU
│       └── selective_inversion.hpp # MUMPS-like methods
│
├── iterative/                       # Iterative solvers
│   ├── iterative_solver_base.hpp   # Iterative base class
│   ├── krylov/
│   │   ├── cg.hpp                  # Conjugate Gradient
│   │   ├── bicgstab.hpp            # BiCGSTAB
│   │   ├── gmres.hpp               # GMRES
│   │   ├── fgmres.hpp              # Flexible GMRES
│   │   ├── minres.hpp              # MINRES
│   │   ├── qmr.hpp                 # Quasi-Minimal Residual
│   │   ├── tfqmr.hpp               # Transpose-Free QMR
│   │   ├── cgs.hpp                 # Conjugate Gradient Squared
│   │   ├── bicg.hpp                # BiConjugate Gradient
│   │   ├── lsqr.hpp                # Least Squares
│   │   ├── craig.hpp               # Craig's method
│   │   └── gcr.hpp                 # Generalized CR
│   ├── stationary/
│   │   ├── jacobi.hpp              # Jacobi iteration
│   │   ├── gauss_seidel.hpp        # Gauss-Seidel
│   │   ├── sor.hpp                 # SOR/SSOR
│   │   ├── richardson.hpp          # Richardson iteration
│   │   └── weighted_jacobi.hpp     # Weighted Jacobi
│   ├── multigrid/
│   │   ├── geometric_mg.hpp        # Geometric multigrid
│   │   ├── algebraic_mg.hpp        # AMG
│   │   ├── smoothers.hpp           # MG smoothers
│   │   ├── coarsening.hpp          # Coarsening strategies
│   │   ├── interpolation.hpp       # MG interpolation
│   │   └── mg_cycles.hpp           # V, W, F cycles
│   ├── domain_decomposition/
│   │   ├── schwarz.hpp             # Schwarz methods
│   │   ├── feti.hpp                # FETI
│   │   ├── bddc.hpp                # BDDC
│   │   └── neumann_neumann.hpp     # Neumann-Neumann
│   └── specialized/
│       ├── chebyshev.hpp           # Chebyshev iteration
│       ├── deflated_cg.hpp         # Deflated CG
│       └── recycling_krylov.hpp    # Recycling methods
│
├── preconditioners/                 # Preconditioning strategies
│   ├── preconditioner_base.hpp     # Preconditioner interface
│   ├── basic/
│   │   ├── identity.hpp            # No preconditioning
│   │   ├── diagonal.hpp            # Jacobi preconditioner
│   │   ├── block_diagonal.hpp      # Block Jacobi
│   │   └── scaling.hpp             # Row/column scaling
│   ├── incomplete/
│   │   ├── ilu.hpp                 # ILU(k)
│   │   ├── ilut.hpp                # ILU with threshold
│   │   ├── icc.hpp                 # Incomplete Cholesky
│   │   ├── ainv.hpp                # Approximate inverse
│   │   └── spai.hpp                # Sparse approximate inverse
│   ├── multigrid/
│   │   ├── amg_preconditioner.hpp  # AMG as preconditioner
│   │   ├── mg_preconditioner.hpp   # Geometric MG
│   │   └── smoothed_aggregation.hpp # SA-AMG
│   ├── domain_decomposition/
│   │   ├── additive_schwarz.hpp    # ASM
│   │   ├── restricted_schwarz.hpp  # RAS
│   │   └── coarse_space.hpp        # Coarse corrections
│   ├── physics_based/
│   │   ├── field_split.hpp         # Field splitting
│   │   ├── block_preconditioner.hpp # Block systems
│   │   ├── schur_complement.hpp    # Schur complements
│   │   └── lsc_preconditioner.hpp  # Least-squares commutator
│   └── advanced/
│       ├── deflation.hpp           # Deflation-based
│       ├── low_rank.hpp            # Low-rank updates
│       └── hierarchical.hpp        # H-matrices
│
├── nonlinear/                       # Nonlinear solvers
│   ├── newton/
│   │   ├── newton_raphson.hpp      # Standard Newton
│   │   ├── modified_newton.hpp     # Modified Newton
│   │   ├── inexact_newton.hpp      # Inexact Newton
│   │   ├── quasi_newton.hpp        # Quasi-Newton (BFGS, L-BFGS)
│   │   └── tensor_newton.hpp       # Tensor methods
│   ├── line_search/
│   │   ├── backtracking.hpp        # Backtracking line search
│   │   ├── wolfe.hpp               # Wolfe conditions
│   │   ├── armijo.hpp              # Armijo rule
│   │   └── polynomial.hpp          # Polynomial line search
│   ├── trust_region/
│   │   ├── trust_region_base.hpp   # Trust region framework
│   │   ├── dogleg.hpp              # Dogleg method
│   │   ├── steihaug.hpp            # Steihaug-Toint
│   │   └── levenberg_marquardt.hpp # Levenberg-Marquardt
│   ├── continuation/
│   │   ├── load_stepping.hpp       # Load control
│   │   ├── arc_length.hpp          # Arc-length method
│   │   ├── displacement_control.hpp # Displacement control
│   │   └── branch_switching.hpp    # Bifurcation tracking
│   ├── fixed_point/
│   │   ├── picard.hpp              # Picard iteration
│   │   ├── anderson.hpp            # Anderson acceleration
│   │   └── aitken.hpp              # Aitken acceleration
│   └── globalization/
│       ├── homotopy.hpp            # Homotopy methods
│       └── parameter_continuation.hpp # Parameter continuation
│
├── eigen/                           # Eigenvalue solvers
│   ├── eigen_solver_base.hpp       # Eigen solver interface
│   ├── dense_eigen/
│   │   ├── qr_algorithm.hpp        # QR algorithm
│   │   ├── divide_conquer.hpp      # Divide and conquer
│   │   └── jacobi_eigen.hpp        # Jacobi method
│   ├── sparse_eigen/
│   │   ├── power_method.hpp        # Power iteration
│   │   ├── inverse_iteration.hpp   # Inverse iteration
│   │   ├── rayleigh_quotient.hpp   # Rayleigh quotient
│   │   ├── arnoldi.hpp             # Arnoldi method
│   │   ├── lanczos.hpp             # Lanczos algorithm
│   │   ├── davidson.hpp            # Davidson method
│   │   └── jacobi_davidson.hpp     # Jacobi-Davidson
│   ├── generalized/
│   │   ├── generalized_eigen.hpp   # A*x = λ*B*x
│   │   ├── shift_invert.hpp        # Shift-and-invert
│   │   └── buckling_eigen.hpp      # Buckling problems
│   └── svd/
│       ├── svd_solver.hpp          # SVD computation
│       └── randomized_svd.hpp      # Randomized SVD
│
├── transient/                       # Time integration
│   ├── time_integrator_base.hpp    # Time integrator interface
│   ├── explicit/
│   │   ├── forward_euler.hpp       # Explicit Euler
│   │   ├── runge_kutta.hpp         # RK methods (RK4, RK45)
│   │   ├── adams_bashforth.hpp     # Adams-Bashforth
│   │   └── central_difference.hpp  # Central difference
│   ├── implicit/
│   │   ├── backward_euler.hpp      # Implicit Euler
│   │   ├── crank_nicolson.hpp      # Crank-Nicolson
│   │   ├── bdf.hpp                 # BDF methods
│   │   ├── newmark.hpp             # Newmark-beta
│   │   ├── hht_alpha.hpp           # HHT-alpha method
│   │   └── generalized_alpha.hpp   # Generalized-alpha
│   ├── adaptive/
│   │   ├── adaptive_timestep.hpp   # Time step control
│   │   ├── embedded_rk.hpp         # Embedded RK
│   │   ├── richardson_extrapolation.hpp
│   │   └── error_control.hpp       # Error estimation
│   └── special/
│       ├── symplectic.hpp          # Symplectic integrators
│       ├── exponential.hpp         # Exponential integrators
│       └── waveform_relaxation.hpp # Waveform iteration
│
├── optimization/                    # Optimization solvers
│   ├── unconstrained/
│   │   ├── gradient_descent.hpp    # Gradient descent
│   │   ├── conjugate_gradient_opt.hpp # Nonlinear CG
│   │   ├── bfgs.hpp                # BFGS
│   │   ├── lbfgs.hpp               # Limited-memory BFGS
│   │   └── trust_region_opt.hpp    # Trust region for optimization
│   ├── constrained/
│   │   ├── sqp.hpp                 # Sequential QP
│   │   ├── interior_point.hpp      # Interior point methods
│   │   ├── augmented_lagrangian.hpp # Augmented Lagrangian
│   │   └── active_set.hpp          # Active set methods
│   └── least_squares/
│       ├── gauss_newton.hpp        # Gauss-Newton
│       └── lm_optimization.hpp     # Levenberg-Marquardt
│
├── saddle_point/                    # Saddle point systems
│   ├── uzawa.hpp                   # Uzawa iteration
│   ├── schur_solver.hpp            # Schur complement approach
│   ├── block_solver.hpp            # Block preconditioners
│   └── constraint_solver.hpp       # Constraint systems
│
├── parallel/                        # Parallel solver infrastructure
│   ├── parallel_krylov.hpp         # Parallel Krylov methods
│   ├── parallel_preconditioner.hpp # Distributed preconditioners
│   ├── communication_avoiding.hpp  # CA-Krylov methods
│   └── gpu_solvers.hpp             # GPU-accelerated solvers
│
├── external/                        # External library interfaces
│   ├── petsc/
│   │   ├── petsc_solver.hpp        # PETSc solver wrapper
│   │   └── petsc_preconditioner.hpp # PETSc preconditioners
│   ├── trilinos/
│   │   ├── aztec_solver.hpp        # AztecOO wrapper
│   │   ├── belos_solver.hpp        # Belos wrapper
│   │   └── ml_preconditioner.hpp   # ML preconditioner
│   ├── hypre/
│   │   ├── hypre_solver.hpp        # Hypre solvers
│   │   └── boomeramg.hpp           # BoomerAMG
│   ├── suitesparse/
│   │   ├── umfpack_solver.hpp      # UMFPACK
│   │   └── cholmod_solver.hpp      # CHOLMOD
│   └── cuda/
│       ├── cusolver.hpp            # cuSOLVER wrapper
│       └── cusparse.hpp            # cuSPARSE wrapper
│
├── utilities/                       # Solver utilities
│   ├── solver_selection.hpp        # Automatic solver selection
│   ├── parameter_tuning.hpp        # Parameter optimization
│   ├── convergence_history.hpp     # Convergence tracking
│   ├── residual_history.hpp        # Residual monitoring
│   └── performance_profiling.hpp   # Performance analysis
│
└── tests/                          # Testing
    ├── unit/                       # Unit tests
    ├── convergence/                # Convergence tests
    ├── performance/                # Performance benchmarks
    └── validation/                 # Validation problems
```

## 🔧 Key Components

### 1. Unified Solver Interface
```cpp
// Base solver interface for all solver types
template<typename Matrix, typename Vector>
class SolverBase {
protected:
    ConvergenceCriteria criteria;
    SolverMonitor monitor;
    SolverStatistics stats;
    
public:
    // Common interface
    virtual SolutionInfo solve(const Matrix& A, 
                              const Vector& b,
                              Vector& x) = 0;
    
    // Configuration
    void set_tolerance(double tol) { 
        criteria.relative_tolerance = tol; 
    }
    
    void set_max_iterations(int max_iter) {
        criteria.max_iterations = max_iter;
    }
    
    // Monitoring
    void attach_monitor(MonitorCallback callback) {
        monitor.add_callback(callback);
    }
};
```

### 2. Krylov Subspace Methods
```cpp
// Flexible GMRES with preconditioning
template<typename Matrix, typename Vector, typename Preconditioner>
class FlexibleGMRES : public IterativeSolver<Matrix, Vector> {
    int restart_size = 30;
    Preconditioner precond;
    
    SolutionInfo solve(const Matrix& A, 
                      const Vector& b, 
                      Vector& x) override {
        int n = b.size();
        int m = restart_size;
        
        // Arnoldi vectors
        std::vector<Vector> V(m + 1, Vector(n));
        
        // Hessenberg matrix
        Matrix H(m + 1, m);
        
        Vector r = b - A * x;
        double beta = norm(r);
        
        while (!converged && iter < max_iter) {
            V[0] = r / beta;
            Vector s(m + 1);
            s[0] = beta;
            
            // Arnoldi process with preconditioning
            for (int j = 0; j < m; ++j) {
                // Apply preconditioner (flexible)
                Vector z = precond.apply(V[j]);
                Vector w = A * z;
                
                // Orthogonalization
                for (int i = 0; i <= j; ++i) {
                    H(i, j) = dot(w, V[i]);
                    w -= H(i, j) * V[i];
                }
                
                H(j + 1, j) = norm(w);
                
                if (H(j + 1, j) < 1e-14) {
                    m = j + 1;
                    break;
                }
                
                V[j + 1] = w / H(j + 1, j);
                
                // Solve least squares problem
                apply_givens_rotations(H, s, j);
                
                // Check convergence
                if (abs(s[j + 1]) < tolerance) {
                    converged = true;
                    break;
                }
            }
            
            // Update solution
            Vector y = solve_upper_triangular(H, s);
            for (int j = 0; j < m; ++j) {
                x += y[j] * precond.apply(V[j]);
            }
            
            // Compute residual for restart
            r = b - A * x;
            beta = norm(r);
            
            iter += m;
        }
        
        return {converged, iter, norm(r)};
    }
};
```

### 3. Algebraic Multigrid
```cpp
// AMG preconditioner/solver
class AlgebraicMultigrid : public Preconditioner {
    struct Level {
        SparseMatrix A;           // Operator at this level
        SparseMatrix P;           // Prolongation
        SparseMatrix R;           // Restriction
        std::unique_ptr<Smoother> smoother;
    };
    
    std::vector<Level> levels;
    
    // Build AMG hierarchy
    void setup(const SparseMatrix& A) {
        levels.clear();
        levels.push_back({A, {}, {}});
        
        // Coarsening loop
        for (int l = 0; l < max_levels; ++l) {
            auto& fine = levels.back();
            
            // Coarsening (e.g., Ruge-Stüben)
            auto C = compute_strength_matrix(fine.A);
            auto splitting = compute_cf_splitting(C);
            
            // Build interpolation
            auto P = build_interpolation(fine.A, splitting);
            
            // Galerkin coarsening
            auto R = P.transpose();
            auto A_coarse = R * fine.A * P;
            
            if (A_coarse.size() < coarse_size) {
                break;  // Coarse enough
            }
            
            levels.push_back({A_coarse, P, R});
        }
        
        // Setup smoothers
        for (auto& level : levels) {
            level.smoother = create_smoother(level.A);
        }
    }
    
    // V-cycle
    Vector apply(const Vector& r) override {
        Vector x(r.size(), 0.0);
        vcycle(0, x, r);
        return x;
    }
    
private:
    void vcycle(int level, Vector& x, const Vector& r) {
        auto& L = levels[level];
        
        if (level == levels.size() - 1) {
            // Coarsest level: direct solve
            x = direct_solve(L.A, r);
            return;
        }
        
        // Pre-smoothing
        L.smoother->apply(L.A, r, x, nu1);
        
        // Compute residual
        Vector res = r - L.A * x;
        
        // Restrict to coarse grid
        Vector r_coarse = levels[level + 1].R * res;
        
        // Solve on coarse grid
        Vector e_coarse(r_coarse.size(), 0.0);
        vcycle(level + 1, e_coarse, r_coarse);
        
        // Prolongate and correct
        x += levels[level + 1].P * e_coarse;
        
        // Post-smoothing
        L.smoother->apply(L.A, r, x, nu2);
    }
};
```

### 4. Newton's Method for Nonlinear Problems
```cpp
// Inexact Newton with line search
template<typename System>
class InexactNewton : public NonlinearSolver {
    LineSearch line_search;
    std::unique_ptr<LinearSolver> linear_solver;
    
    Solution solve(System& system, Solution& u) override {
        for (int iter = 0; iter < max_iter; ++iter) {
            // Evaluate residual
            Vector F = system.compute_residual(u);
            double res_norm = norm(F);
            
            // Check convergence
            if (res_norm < tolerance) {
                return {true, iter, res_norm};
            }
            
            // Compute Jacobian
            Matrix J = system.compute_jacobian(u);
            
            // Solve Newton system (inexactly)
            Vector delta_u;
            double eta = compute_forcing_term(res_norm);
            linear_solver->set_tolerance(eta * res_norm);
            linear_solver->solve(J, -F, delta_u);
            
            // Line search
            double alpha = line_search.compute(
                [&](const Vector& v) { 
                    return norm(system.compute_residual(v)); 
                },
                u, delta_u, F
            );
            
            // Update solution
            u += alpha * delta_u;
            
            // Monitor convergence
            monitor.report(iter, res_norm, alpha);
        }
        
        return {false, max_iter, norm(F)};
    }
};
```

### 5. Time Integration
```cpp
// Generalized-alpha method for dynamics
class GeneralizedAlpha : public TimeIntegrator {
    double alpha_m, alpha_f, beta, gamma;
    
    void step(DynamicSystem& system,
             State& state,
             double dt) override {
        // Predictors
        Vector u_pred = state.u + dt * state.v 
                      + dt*dt * ((0.5 - beta) * state.a);
        Vector v_pred = state.v + dt * ((1 - gamma) * state.a);
        
        // Solve nonlinear system at intermediate time
        auto residual = [&](const Vector& a_new) {
            Vector u_af = (1 - alpha_f) * u_pred + alpha_f * state.u
                        + beta * dt * dt * a_new;
            Vector v_af = v_pred + gamma * dt * a_new;
            Vector a_am = (1 - alpha_m) * a_new + alpha_m * state.a;
            
            Matrix M = system.mass_matrix();
            Matrix C = system.damping_matrix();
            Matrix K = system.stiffness_matrix(u_af);
            Vector F = system.external_force(t + alpha_f * dt);
            
            return M * a_am + C * v_af + K * u_af - F;
        };
        
        // Newton solve for acceleration
        Vector a_new = newton_solve(residual, state.a);
        
        // Update state
        state.u = u_pred + beta * dt * dt * a_new;
        state.v = v_pred + gamma * dt * a_new;
        state.a = a_new;
        state.t += dt;
    }
};
```

### 6. Field-Split Preconditioner
```cpp
// Block preconditioner for multi-physics
class FieldSplitPreconditioner : public Preconditioner {
    struct Field {
        std::string name;
        IndexSet indices;
        std::unique_ptr<Preconditioner> precond;
    };
    
    std::vector<Field> fields;
    SchurComplementType schur_type;
    
    Vector apply(const Vector& r) override {
        if (schur_type == SchurComplementType::NONE) {
            // Block diagonal or triangular
            return apply_additive(r);
        } else {
            // Schur complement approach
            return apply_schur(r);
        }
    }
    
private:
    Vector apply_schur(const Vector& r) {
        // For 2x2 block system:
        // [A  B] [x1]   [r1]
        // [C  D] [x2] = [r2]
        
        auto r1 = extract(r, fields[0].indices);
        auto r2 = extract(r, fields[1].indices);
        
        // Step 1: Solve A*y1 = r1
        Vector y1 = fields[0].precond->apply(r1);
        
        // Step 2: Compute Schur complement RHS
        Vector s = r2 - C * y1;
        
        // Step 3: Solve Schur complement system
        Vector x2 = solve_schur_complement(s);
        
        // Step 4: Back-substitute
        Vector x1 = fields[0].precond->apply(r1 - B * x2);
        
        return combine(x1, x2);
    }
};
```

### 7. GPU-Accelerated Solver
```cpp
// GPU Krylov solver
class GPUConjugateGradient : public IterativeSolver {
    void solve_gpu(const DeviceMatrix& A,
                  const DeviceVector& b,
                  DeviceVector& x) {
        int n = b.size();
        DeviceVector r(n), p(n), Ap(n);
        
        // Initial residual
        cublas_dgemv(handle, -1.0, A, x, 1.0, b, r);
        p = r;
        
        double rsold = cublas_ddot(handle, r, r);
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // Matrix-vector product
            cusparse_dcsrmv(handle, A, p, Ap);
            
            double alpha = rsold / cublas_ddot(handle, p, Ap);
            
            // Update solution and residual
            cublas_daxpy(handle, alpha, p, x);
            cublas_daxpy(handle, -alpha, Ap, r);
            
            double rsnew = cublas_ddot(handle, r, r);
            
            if (sqrt(rsnew) < tolerance) {
                converged = true;
                break;
            }
            
            double beta = rsnew / rsold;
            
            // Update search direction
            cublas_dscal(handle, beta, p);
            cublas_daxpy(handle, 1.0, r, p);
            
            rsold = rsnew;
        }
    }
};
```

## Performance Optimizations

### Solver Selection Strategy
```cpp
class AutoSolver {
    std::unique_ptr<LinearSolver> select_solver(
        const SparseMatrix& A,
        const ProblemCharacteristics& props) {
        
        if (A.is_symmetric()) {
            if (A.is_positive_definite()) {
                if (A.size() < 10000) {
                    return make_cholesky_solver();
                } else {
                    return make_cg_solver();
                }
            } else {
                return make_minres_solver();
            }
        } else {
            if (A.condition_number() < 1e6) {
                return make_gmres_solver(30);
            } else {
                return make_gmres_solver(50)
                    ->with_preconditioner(make_ilu());
            }
        }
    }
};
```

## Integration Points

### With numeric/
- Uses sparse/dense matrices from numeric
- Leverages BLAS operations for performance
- Utilizes graph algorithms for AMG

### With assembly/
- Receives assembled system matrices
- Works with constraint-modified systems
- Handles block-structured matrices

### With physics/
- Solves linearized physics equations
- Handles multi-physics coupling
- Manages nonlinear material models

### With adaptation/
- Provides error estimates for adaptivity
- Re-solves on adapted meshes
- Transfers solution between grids

## Success Metrics

1. **Direct Solver Speed**: O(n²) for banded, O(n^1.5) for sparse
2. **Iterative Convergence**: < 100 iterations for well-conditioned
3. **AMG Setup**: < 10% of solve time
4. **Newton Convergence**: Quadratic near solution
5. **Parallel Efficiency**: > 80% weak scaling
6. **GPU Speedup**: > 10x for large sparse systems

## Key Features

1. **Complete Coverage**: Direct, iterative, eigenvalue, and nonlinear
2. **Preconditioner-Rich**: Extensive preconditioning options
3. **Physics-Aware**: Specialized solvers for different physics
4. **External Integration**: Seamless use of PETSc, Trilinos, etc.
5. **GPU-Ready**: Native GPU implementations
6. **Adaptive**: Automatic solver selection and parameter tuning

This architecture provides comprehensive solution capabilities for all types of finite element systems, from small dense problems to massive distributed simulations.