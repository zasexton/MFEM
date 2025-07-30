/**
 * @file constants.h
 * @author Finite Element MultiPhysics Solver
 * @brief Compile‑time mathematical (and selected physical) constants.
 *
 * This header is part of the **numeric::scalar** sub‑library.  All
 * constants are provided as `constexpr` variable templates so they can be
 * used in constant‑evaluation contexts and specialised at compile time for
 * different floating‑point precisions.
 *
 * ```cpp
 * using namespace numeric::scalar;
 * auto angle  = 30.0 * pi_v<double> / 180.0;   // OK at compile‑time
 * auto speed  = 3.0 * c_v<float>;              // physical constant
 * ```
 *
 * The default instantiation (`<long double>`) yields the highest precision
 * available on the target compiler.  When a concrete type is required (e.g.
 * GPU kernels restricted to `float`) the template parameter can be supplied
 * explicitly: `pi_v<float>`.
 *
 * For physical constants please refer to the 2022 NIST source:
 * https://physics.nist.gov/cuu/Constants/index.html
 *
 * @ingroup numeric_scalar
 */

#pragma once

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <numbers>
#include <concepts>

namespace numeric::scalar::constants {
    template <typename T>
    concept Float = std::floating_point<T>;
    /// @addtogroup math_constants
    /// @{

    /** π (pi) – ratio of a circle's circumference to its diameter. */
    template <Float T>
    inline constexpr T pi               = std::numbers::pi_v<T>;

    /** e (Euler's number) – base of natural logarithms. */
    template <Float T>
    inline constexpr T e                = std::numbers::e_v<T>;

    /** √2 – square root of two. */
    template <Float T>
    inline constexpr T sqrt2            = std::numbers::sqrt2_v<T>;

    /** φ (golden ratio) – (1+√5)/2. */
    template <Float T>
    inline constexpr T golden_ratio     = std::numbers::phi_v<T>;

    /** ln10 – natural logarithm of ten. */
    template <Float T>
    inline constexpr T ln10             = std::numbers::ln10_v<T>;

    /** ln2 – natural logarithm of two. */
    template <Float T>
    inline constexpr T ln2              = std::numbers::ln2_v<T>;

    /** log₁₀e – base‑10 logarithm of Euler’s number. */
    template <Float T>
    inline constexpr T log10e           = std::numbers::log10e_v<T>;

    /** log₂e – base‑2 logarithm of Euler’s number. */
    template <Float T>
    inline constexpr T log2e            = std::numbers::log2e_v<T>;

    // ---------------------------------------------------------------------------
    // Universal physical constants — CODATA 2022 (or exact SI) values
    // ---------------------------------------------------------------------------

    /** Speed of light in vacuum (m s⁻¹). Exact by SI definition.
     * Fundamental upper speed limit of the Universe and the basis for the SI definition of the metre.*/
    template <Float T>
    inline constexpr T speed_of_light_vacuum = static_cast<T>(299'792'458.0L);

    /** Planck constant (J s). Exact by SI definition.
     * Quantum of action that links energy and frequency (*E=hν*); cornerstone of quantum mechanics.*/
    template <Float T>
    inline constexpr T planck_constant = static_cast<T>(6.626'070'15e-34L);

    /** Reduced Planck constant (ħ=h/2π, J s). Exact by SI definition.
     * Often appears in angular‑frequency formulations of quantum mechanics. */
    template <Float T>
    inline constexpr T reduced_planck_constant = static_cast<T>(1.054'571'817e-34L);

    /** Vacuum magnetic permeability (μ₀, N A⁻²⁻). Exact by SI definition.
     * Relates magnetic field to current in free space; derived from *c* and the fine‑structure constant.*/
    template <Float T>
    inline constexpr T vacuum_magnetic_permeability = static_cast<T>(1.256'637'061'27e-6L);

    /** Vacuum permittivity (ε₀=1/(μ₀c²), F m⁻¹).
     * Relates electric displacement to field in free space. */
    template <Float T>
    inline constexpr T vacuum_electric_permittivity = static_cast<T>(8.854'187'8188e-12L);

    /** Newtonian gravitational constant (G, m³ kg⁻¹ s⁻²).
     * Sets the strength of classical gravity between masses. */
    template <Float T>
    inline constexpr T newtonian_gravitational_constant = static_cast<T>(6.674'30e-11L);

    /** Planck length (m).
     *  Characteristic length at which quantum‑gravity effects become significant.*/
    template <Float T>
    inline constexpr T planck_length = static_cast<T>(1.616'255e-35L);

    /** Planck mass (kg).
     *  Mass scale where a particle’s Compton wavelength equals its Schwarzschild radius.*/
    template <Float T>
    inline constexpr T planck_mass = static_cast<T>(2.176'434e-8L);

    /** Planck time (s).
     *  Time taken for light to travel one Planck length in vacuum.*/
    template <Float T>
    inline constexpr T planck_time = static_cast<T>(5.391'247e-44L);

    /** Planck Temperature (K).
     *  Theoretical upper bound temperature beyond which known physics breaks down.*/
    template <Float T>
    inline constexpr T planck_temp = static_cast<T>(1.416'784e32L);

    // ---------------------------------------------------------------------------
    // Defined constants — CODATA 2022 (or exact SI) values
    // ---------------------------------------------------------------------------

    /** Avogadro constant (mol⁻¹). Exact by SI definition.
     *  Exact count of entities (atoms, molecules, …) in one mole of substance. */
    template <Float T>
    inline constexpr T avogadro = static_cast<T>(6.022'140'76e23L);

    /** Boltzmann constant (J K⁻¹). Exact by SI definition.
     *  Converts temperature to thermal energy (*E=k_BT*). */
    template <Float T>
    inline constexpr T boltzmann = static_cast<T>(1.380'649e-23L);

    /** Elementary charge (C). Exact by SI definition.
     *  Magnitude of electric charge carried by a proton (electron has −*e*). */
    template <Float T>
    inline constexpr T elementary_charge = static_cast<T>(1.602'176'634e-19L);

    /** Hyperfine transition frequency of Cs-133 (Hz). Exact by SI definition.
     *  Precisely 9,192,631,770 Hz defines the SI second.*/
    template <Float T>
    inline constexpr T Cs_frequency = static_cast<T>(9'192'631'770L);

    /** Luminous efficacy (lm W⁻¹). Exact by SI definition.
     *  683 lm W⁻¹ for monochromatic light at 540 THz defines the candela*/
    template <Float T>
    inline constexpr T luminous_efficacy = static_cast<T>(683L);

    /** Standard acceleration of gravity (m s⁻²). Exact by SI definition.
     *  Conventional reference value for Earth’s surface gravity. */
    template <Float T>
    inline constexpr T standard_gravity = static_cast<T>(9.806'65L);

    /** Standard Atmosphere (Pa). Exact by SI definition.
     *  Conventional reference pressure equal to 101,325 Pa.*/
    template <Float T>
    inline constexpr T atm = static_cast<T>(101'325L);

    /** Standard-State Pressure (Pa). Exact by SI definition.
     *  IUPAC reference pressure of 100kPa used in thermodynamics.*/
    template <Float T>
    inline constexpr T ssp = static_cast<T>(100'000L);

    // ---------------------------------------------------------------------------
    // Electromagnetic constants — CODATA 2022 (or exact SI) values
    // ---------------------------------------------------------------------------

    /** Bohr magneton ( $\mu_{\rm B} = e\hbar/2m_{\rm e}$, J T⁻¹).
     *  Magnetic dipole moment of an electron due to its orbital or spin angular momentum.*/
    template <Float T>
    inline constexpr T bohr_magneton = static_cast<T>(9.274'010'0657e-24L);

    /** Conductance quantum ( $G_0 = 2e^2/2\rmpi\hbar$, S). Exact by SI definition.
     *  Fundamental quantum unit of electrical conductance. */
    template <Float T>
    inline constexpr T conductance_quantum = static_cast<T>(7.748'091'729e-5L);

    /** Conventrional Value of Ampere-90 ( $A_{90} = (K_{{\rm J}-90}R_{{\rm K}-90}/K_{\rm J}R_{\rm K})\,{\rm A}$, A). Exact by SI definition.
     *  Legacy 1990‑2019 conventional value of the ampere used in electrical metrology. */
    template <Float T>
    inline constexpr T ampere_90 = static_cast<T>(1.000'000'088'87L);

    /** Conventrional Value of Coulomb-90 ( $C_{90} = (K_{{\rm J}-90}R_{{\rm K}-90}/K_{\rm J}R_{\rm K})\,{\rm C}$, C) Exact by SI definition.
     *  Legacy conventional value of charge consistent with A₉₀*/
    template <Float T>
    inline constexpr T coulomb_90 = static_cast<T>(1.000'000'088'87L);

    /** Conventional Value of Farad-90 ( $F_{90} = (R_{{\rm K}-90}/R_{{\rm K}})\,{\rm F}$, F). Exact by SI definition.
     *  Legacy conventional value for capacitance.*/
    template <Float T>
    inline constexpr T farad_90 = static_cast<T>(0.999'999'982'20L);

    /** Conventrional Value of Henry-90 ( $H_{90} = (R_{\rm K}/R_{{\rm K}-90})\,{\rm H}$, H). Exact by SI definition.
     *  Legacy conventional value for inductance.*/
    template <Float T>
    inline constexpr T henry_90 = static_cast<T>(1.000'000'017'79L);

    /** Conventional Value of Josephson Constant ( $K_{J - 90}, Hz V⁻¹). Exact by SI definition.
     *  Legacy value linking frequency and voltage via the Josephson effect.*/
    template <Float T>
    inline constexpr T josephson_90 = static_cast<T>(483'597.9e9L);

    /** Conventional Value of Ohm-90 ( ${\it \Omega}_{90} = (R_{\rm K}/R_{{\rm K}-90})\,{\rm \Omega}$, Ω). Exact by SI definition.
     *  Legacy conventional value for resistance.*/
    template <Float T>
    inline constexpr T ohm_90 = static_cast<T>(1.000'000'017'79L);

    /** Conventional Value of Volt-90 ( $V_{90} = (K_{{\rm J}-90}/K_{\rm J})\,{\rm V}$, V). Exact by SI definition.
     *  Legacy conventional value for voltage.*/
    template <Float T>
    inline constexpr T volt_90 = static_cast<T>(1.000'000'106'66L);

    /** Conventional Value of von Klitzing Constant ($R_{K-90}$, Ω). Exact by SI definition.
     *  Quantum Hall resistance used in conventional (1990‑2019) electrical standards.*/
    template <Float T>
    inline constexpr T von_klitzing_90 = static_cast<T>(25'812.807L);

    /** Conventrional Value of Watt-90 ( $W_{90} = (K^2_{{\rm J}-90}R_{{\rm K}-90}/K^2_{\rm J}R_{\rm K})\,{\rm W}$, W). Exact by SI definition.
     *  Legacy conventional value for power used with electrical standards. */
    template <Float T>
    inline constexpr T watt_90 = static_cast<T>(1.000'000'195'53L);

    /** Josephson Constant ($K_{J}$, Hz V⁻¹). Exact by SI definition.
     *  Exact modern value relating frequency to voltage in the Josephson effect.*/
    template <Float T>
    inline constexpr T josephson = static_cast<T>(483'597.848'4e9L);

    /** Magnetic Flux Quantum ( ${\it\Phi}_0 = 2\rmpi \hbar/2e$, Wb). Exact by SI definition.
     *  Smallest quantum of magnetic flux passing through a superconducting loop.*/
    template <Float T>
    inline constexpr T magnetic_flux_quantum = static_cast<T>(2.067'833e-15L);

    /** Nuclear Magneton ($\mu_{\rm N} = e\hbar/2m_{\rm p}$, J T⁻¹).
     *  Magnetic moment scale for nucleons such as the proton.*/
    template <Float T>
    inline constexpr T nuclear_magneton = static_cast<T>(5.050'783'7393e-27L);

    /** von Klitzing Constant ($R_{\rm K} = \mu_0 c/2\alpha= 2\rmpi\hbar/e^2$, Ω). Exact by SI definition.
     *  Exact quantum Hall resistance value 25 812.807 45 Ω used in modern SI.*/
    template <Float T>
    inline constexpr T von_klitzing = static_cast<T>(25'812.807'45);

    // ---------------------------------------------------------------------------
    // Atomic & Nuclear constants — CODATA 2022 (or exact SI) values
    // ---------------------------------------------------------------------------

    // To be added later.

} // namespace numeric::scalar::constants
#endif //CONSTANTS_H
