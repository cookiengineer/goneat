// Package vector provides small, allocation-free numeric kernels used by the
// goneat engine.
//
// The package has two backends:
//
//   - a portable scalar implementation that builds everywhere, and
//   - a SIMD implementation compiled when the portable "simd" package is
//     available (GOEXPERIMENT=simd, Go 1.27+).
//
// Both backends are selected at compile time via build constraints and expose
// exactly the same API, so callers never need to care which one is active.
// Kernels operate element-wise on float64 slices and are intended for the hot
// paths of network activation and backpropagation.
package vector
