#pragma once

// Placeholder: Generic shock-capturing terms for FEM formulations.
//
// This header represents a shared location for shock-capturing
// techniques (e.g., entropy viscosity, slope/flux limiters, TVB/TVB-like
// strategies) that can be applied across multiple PDEs (compressible
// flow, shallow water, scalar conservation laws, etc.). Physics modules
// should configure and consume these terms; implementations live here in
// fem/ so they are reusable across domains.
//
// API sketch (to be defined):
//   - assemble_shock_capturing(FormContext&, const Residuals&, Matrix& Ke, Vector& Re)
//   - compute_entropy_viscosity(...)
//   - limiter interfaces for DG/CG formulations

