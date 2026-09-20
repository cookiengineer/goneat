package adapter

import (
	"fmt"
	"math"
)

// ComplexAdapter vectorizes a complex value as two dimensions (real and
// imaginary). By default each part is squashed with tanh, which keeps
// unbounded values inside [-1, 1].
type ComplexAdapter struct {
	name         string
	realScale    float64
	imagScale    float64
	value        complex128
	placement    Point
	hasPlacement bool
}

// NewComplex creates a complex adapter with separate scales for the real and
// imaginary parts. A scale of 0 defaults to 1.
func NewComplex(name string, realScale, imagScale float64) *ComplexAdapter {
	if realScale == 0 {
		realScale = 1
	}
	if imagScale == 0 {
		imagScale = 1
	}
	return &ComplexAdapter{name: name, realScale: realScale, imagScale: imagScale}
}

// NewComplexUniform creates a complex adapter using the same scale for both
// parts.
func NewComplexUniform(name string, scale float64) *ComplexAdapter {
	return NewComplex(name, scale, scale)
}

// WithPlacement assigns a substrate placement coordinate.
func (a *ComplexAdapter) WithPlacement(point Point) *ComplexAdapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *ComplexAdapter) Name() string { return a.name }

// Placement implements Placed.
func (a *ComplexAdapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions is always 2 (real, imaginary).
func (a *ComplexAdapter) Dimensions() int { return 2 }

// SetValue sets the current complex value.
func (a *ComplexAdapter) SetValue(value complex128) { a.value = value }

// Value returns the current complex value.
func (a *ComplexAdapter) Value() complex128 { return a.value }

// Sense encodes the real and imaginary parts.
func (a *ComplexAdapter) Sense() []float64 {
	return []float64{
		math.Tanh(real(a.value) / a.realScale),
		math.Tanh(imag(a.value) / a.imagScale),
	}
}

// Control decodes a normalized pair into a complex value.
func (a *ComplexAdapter) Control(values []float64) error {
	if len(values) < 2 {
		return fmt.Errorf("adapter: complex %q expects 2 values, got %d", a.name, len(values))
	}
	a.value = complex(atanhSquash(values[0])*a.realScale, atanhSquash(values[1])*a.imagScale)
	return nil
}
