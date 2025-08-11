#pragma once

#ifndef BASE_INTERFACE_H
#define BASE_INTERFACE_H

#include <type_traits>
#include <concepts>
#include <string>
#include <memory>

namespace fem::core::base {

/**
 * @brief Pure virtual interface base class
 *
 * Provides a lightweight base for defining interfaces without
 * the overhead of the full Object hierarchy. Use this when you need
 * pure interfaces that multiple unrelated classes can implement.
 *
 * Features:
 * - Pure virtual destructor (forces derived classes to be abstract)
 * - No data members (zero overhead)
 * - No vtable overhead beyond the destructor
 * - Designed for interface segregation principle
 *
 * Use cases:
 * - Plugin interfaces
 * - Algorithm strategy interfaces
 * - Cross-module communication contracts
 * - Third-party library integration points
 */
    class Interface {
    public:
        /**
         * @brief Pure virtual destructor
         * Makes this class abstract and ensures proper cleanup
         */
        virtual ~Interface() = 0;

    protected:
        /**
         * @brief Protected default constructor
         * Only derived classes can construct
         */
        Interface() = default;

        /**
         * @brief Protected copy constructor
         */
        Interface(const Interface&) = default;

        /**
         * @brief Protected copy assignment
         */
        Interface& operator=(const Interface&) = default;

        /**
         * @brief Protected move constructor
         */
        Interface(Interface&&) = default;

        /**
         * @brief Protected move assignment
         */
        Interface& operator=(Interface&&) = default;
    };

// Pure virtual destructor must be defined
    inline Interface::~Interface() = default;

/**
 * @brief CRTP Interface helper for type-safe interfaces
 *
 * Provides compile-time interface checking and type safety.
 * Use when you want to ensure interfaces are properly implemented
 * and catch errors at compile time.
 *
 * Usage:
 *   class MyInterface : public TypedInterface<MyInterface> {
 *   public:
 *       virtual void required_method() = 0;
 *   };
 */
    template<typename Derived>
    class TypedInterface : public Interface {
    public:
        /**
         * @brief Get the interface type name
         */
        virtual std::string interface_name() const {
            return typeid(Derived).name();
        }

        /**
         * @brief Check if this object implements a specific interface
         */
        template<typename InterfaceType>
        [[nodiscard]] bool implements() const noexcept {
            return dynamic_cast<const InterfaceType*>(this) != nullptr;
        }

        /**
         * @brief Safe cast to interface type
         */
        template<typename InterfaceType>
        [[nodiscard]] InterfaceType* as_interface() noexcept {
            return dynamic_cast<InterfaceType*>(this);
        }

        template<typename InterfaceType>
        [[nodiscard]] const InterfaceType* as_interface() const noexcept {
            return dynamic_cast<const InterfaceType*>(this);
        }

    protected:
        TypedInterface() = default;
        ~TypedInterface() override = default;
    };

// === Common FEM Interfaces ===

/**
 * @brief Interface for objects that can be calculated/computed
 */
    class ICalculable : public TypedInterface<ICalculable> {
    public:
        /**
         * @brief Perform calculation
         */
        virtual void calculate() = 0;

        /**
         * @brief Check if calculation is required
         */
        virtual bool needs_calculation() const = 0;

        /**
         * @brief Reset calculation state
         */
        virtual void reset_calculation() = 0;
    };

/**
 * @brief Interface for objects that can be serialized
 */
    class ISerializable : public TypedInterface<ISerializable> {
    public:
        /**
         * @brief Serialize object to string
         */
        virtual std::string serialize() const = 0;

        /**
         * @brief Deserialize object from string
         */
        virtual bool deserialize(const std::string& data) = 0;

        /**
         * @brief Get serialization version
         */
        virtual int get_version() const = 0;
    };

/**
 * @brief Interface for objects that can be validated
 */
    class IValidatable : public TypedInterface<IValidatable> {
    public:
        /**
         * @brief Validate object state
         */
        virtual bool is_valid() const = 0;

        /**
         * @brief Get validation errors
         */
        virtual std::vector<std::string> get_validation_errors() const = 0;

        /**
         * @brief Attempt to fix validation errors
         */
        virtual bool try_fix_errors() = 0;
    };

/**
 * @brief Interface for objects that can be cloned
 */
    class ICloneable : public TypedInterface<ICloneable> {
    public:
        /**
         * @brief Create a deep copy of this object
         */
        virtual std::unique_ptr<ICloneable> clone() const = 0;

        /**
         * @brief Create a shallow copy of this object
         */
        virtual std::unique_ptr<ICloneable> shallow_clone() const {
            // Default implementation delegates to deep clone
            return clone();
        }
    };

/**
 * @brief Interface for objects that can be configured
 */
    class IConfigurable : public TypedInterface<IConfigurable> {
    public:
        /**
         * @brief Configure object from parameters
         */
        virtual bool configure(const std::unordered_map<std::string, std::string>& params) = 0;

        /**
         * @brief Get current configuration
         */
        virtual std::unordered_map<std::string, std::string> get_configuration() const = 0;

        /**
         * @brief Get list of supported configuration keys
         */
        virtual std::vector<std::string> get_supported_keys() const = 0;

        /**
         * @brief Reset to default configuration
         */
        virtual void reset_configuration() = 0;
    };

/**
 * @brief Interface for objects that can provide progress updates
 */
    class IProgressReporter : public TypedInterface<IProgressReporter> {
    public:
        /**
         * @brief Get current progress (0.0 to 1.0)
         */
        virtual double get_progress() const = 0;

        /**
         * @brief Get progress description
         */
        virtual std::string get_progress_description() const = 0;

        /**
         * @brief Check if operation is complete
         */
        virtual bool is_complete() const = 0;

        /**
         * @brief Check if operation was cancelled
         */
        virtual bool is_cancelled() const = 0;

        /**
         * @brief Cancel the operation
         */
        virtual void cancel() = 0;
    };

/**
 * @brief Interface for objects that can be compared
 */
    template<typename T>
    class IComparable : public TypedInterface<IComparable<T>> {
    public:
        /**
         * @brief Compare with another object
         * @return < 0 if this < other, 0 if equal, > 0 if this > other
         */
        virtual int compare(const T& other) const = 0;

        /**
         * @brief Check equality
         */
        virtual bool equals(const T& other) const {
            return compare(other) == 0;
        }

        /**
         * @brief Less than comparison
         */
        bool operator<(const T& other) const {
            return compare(other) < 0;
        }

        /**
         * @brief Equality comparison
         */
        bool operator==(const T& other) const {
            return equals(other);
        }

        /**
         * @brief Inequality comparison
         */
        bool operator!=(const T& other) const {
            return !equals(other);
        }
    };

// === Solver-Specific Interfaces ===

/**
 * @brief Interface for linear system solvers
 */
    class ILinearSolver : public TypedInterface<ILinearSolver> {
    public:
        /**
         * @brief Solve linear system Ax = b
         */
        virtual bool solve(const class Matrix& A, const class Vector& b, class Vector& x) = 0;

        /**
         * @brief Set solver tolerance
         */
        virtual void set_tolerance(double tolerance) = 0;

        /**
         * @brief Get solver tolerance
         */
        virtual double get_tolerance() const = 0;

        /**
         * @brief Get number of iterations from last solve
         */
        virtual int get_iterations() const = 0;

        /**
         * @brief Get residual from last solve
         */
        virtual double get_residual() const = 0;
    };

/**
 * @brief Interface for nonlinear solvers
 */
    class INonlinearSolver : public TypedInterface<INonlinearSolver> {
    public:
        /**
         * @brief Solve nonlinear system F(x) = 0
         */
        virtual bool solve(std::function<void(const class Vector&, class Vector&)> residual_func,
                           std::function<void(const class Vector&, class Matrix&)> jacobian_func,
                           class Vector& x) = 0;

        /**
         * @brief Set maximum iterations
         */
        virtual void set_max_iterations(int max_iter) = 0;

        /**
         * @brief Get maximum iterations
         */
        virtual int get_max_iterations() const = 0;

        /**
         * @brief Set convergence tolerance
         */
        virtual void set_tolerance(double tolerance) = 0;

        /**
         * @brief Check if last solve converged
         */
        virtual bool converged() const = 0;
    };

/**
 * @brief Generic interface for time integration schemes
 *
 * Designed to handle any time-dependent ODE/PDE system:
 * - Structural dynamics (displacement, velocity, acceleration)
 * - Heat transfer (temperature, temperature rate)
 * - Fluid flow (velocity, pressure)
 * - Coupled multi-physics problems
 */
    class ITimeIntegrator : public TypedInterface<ITimeIntegrator> {
    public:
        /**
         * @brief Advance solution by one time step
         * @param dt Time step size
         * @param state_vectors Vector of state variables (e.g., {u, v, a} for dynamics)
         * @param time_derivatives Vector of time derivatives (e.g., {v, a, a_dot})
         * @return true if step was successful
         */
        virtual bool step(double dt,
                          std::vector<class Vector*>& state_vectors,
                          std::vector<class Vector*>& time_derivatives) = 0;

        /**
         * @brief Set time step size
         */
        virtual void set_time_step(double dt) = 0;

        /**
         * @brief Get current time
         */
        virtual double get_current_time() const = 0;

        /**
         * @brief Reset integrator state
         */
        virtual void reset(double initial_time = 0.0) = 0;

        /**
         * @brief Get integration order (1st, 2nd, etc.)
         */
        virtual int get_order() const = 0;

        /**
         * @brief Check if integrator is implicit
         */
        virtual bool is_implicit() const = 0;

        /**
         * @brief Get number of state vectors required
         */
        virtual size_t get_required_state_count() const = 0;
    };

/**
 * @brief Interface for mesh generators
 */
    class IMeshGenerator : public TypedInterface<IMeshGenerator> {
    public:
        /**
         * @brief Generate mesh for given geometry
         */
        virtual bool generate_mesh(const class Geometry& geometry, class Mesh& mesh) = 0;

        /**
         * @brief Set mesh density/refinement level
         */
        virtual void set_refinement_level(int level) = 0;

        /**
         * @brief Set element type to generate
         */
        virtual void set_element_type(const std::string& element_type) = 0;

        /**
         * @brief Get estimated number of elements
         */
        virtual size_t estimate_element_count(const class Geometry& geometry) const = 0;
    };

/**
 * @brief Interface for material models
 */
    class IMaterialModel : public TypedInterface<IMaterialModel> {
    public:
        /**
         * @brief Compute stress from strain
         */
        virtual void compute_stress(const class Strain& strain, class Stress& stress) = 0;

        /**
         * @brief Compute material tangent matrix
         */
        virtual void compute_tangent(const class Strain& strain, class Matrix& tangent) = 0;

        /**
         * @brief Update material state
         */
        virtual void update_state(const class Strain& strain) = 0;

        /**
         * @brief Reset material to initial state
         */
        virtual void reset_state() = 0;

        /**
         * @brief Check if material model is nonlinear
         */
        virtual bool is_nonlinear() const = 0;
    };

// === Interface Checking Utilities ===

/**
 * @brief Concept to check if a type implements an interface
 */
    template<typename T, typename InterfaceType>
    concept ImplementsInterface = std::is_base_of_v<InterfaceType, T> ||
    requires(T t) {
{ dynamic_cast<InterfaceType*>(&t) } -> std::convertible_to<InterfaceType*>;
};

/**
 * @brief Check if object implements multiple interfaces
 */
template<typename T, typename... Interfaces>
concept ImplementsInterfaces = (ImplementsInterface<T, Interfaces> && ...);

/**
 * @brief Utility to safely cast to interface and call method
 */
template<typename InterfaceType, typename ObjectType>
auto safe_interface_call(ObjectType* obj, auto method, auto... args) -> std::optional<decltype(method(args...))> {
    if (auto* interface = dynamic_cast<InterfaceType*>(obj)) {
        return method(interface, args...);
    }
    return std::nullopt;
}

/**
 * @brief Helper macro to define interface method forwarding
 */
#define FEM_IMPLEMENT_INTERFACE_METHOD(InterfaceType, method_name) \
    auto method_name() const -> decltype(static_cast<const InterfaceType*>(this)->method_name()) { \
        return static_cast<const InterfaceType*>(this)->method_name(); \
    } \
    auto method_name() -> decltype(static_cast<InterfaceType*>(this)->method_name()) { \
        return static_cast<InterfaceType*>(this)->method_name(); \
    }

} // namespace fem::core

#endif //BASE_INTERFACE_H
