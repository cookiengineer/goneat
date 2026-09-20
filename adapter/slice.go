package adapter

import "fmt"

// SliceAdapter vectorizes a fixed-length []float64 as one dimension per
// element. Each element is divided by its scale and clamped, which lets a
// single adapter expose a whole observation vector (for example, a row of
// pixels or a set of joint angles). The adapter owns the current slice.
type SliceAdapter struct {
	name         string
	scales       []float64
	value        []float64
	placement    Point
	hasPlacement bool
}

// NewSlice creates a slice adapter with the given number of dimensions and a
// single scale applied to every element. A scale of 0 defaults to 1.
func NewSlice(name string, dimensions int, scale float64) *SliceAdapter {
	scales := make([]float64, dimensions)
	for i := range scales {
		if scale == 0 {
			scales[i] = 1
		} else {
			scales[i] = scale
		}
	}
	return NewSliceWithScales(name, scales)
}

// NewSliceWithScales creates a slice adapter with a per-element scale.
func NewSliceWithScales(name string, scales []float64) *SliceAdapter {
	copied := make([]float64, len(scales))
	for i, scale := range scales {
		if scale == 0 {
			scale = 1
		}
		copied[i] = scale
	}
	return &SliceAdapter{
		name:   name,
		scales: copied,
		value:  make([]float64, len(copied)),
	}
}

// WithPlacement assigns a substrate placement coordinate.
func (a *SliceAdapter) WithPlacement(point Point) *SliceAdapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *SliceAdapter) Name() string { return a.name }

// Placement implements Placed.
func (a *SliceAdapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions returns the fixed slice length.
func (a *SliceAdapter) Dimensions() int { return len(a.scales) }

// SetValues replaces the current slice. The length must match the adapter.
func (a *SliceAdapter) SetValues(values []float64) error {
	if len(values) != len(a.scales) {
		return fmt.Errorf("adapter: slice %q expects %d values, got %d", a.name, len(a.scales), len(values))
	}
	copy(a.value, values)
	return nil
}

// Values returns a copy of the current slice.
func (a *SliceAdapter) Values() []float64 { return append([]float64(nil), a.value...) }

// Sense encodes the slice.
func (a *SliceAdapter) Sense() []float64 {
	encoded := make([]float64, len(a.scales))
	for i, scale := range a.scales {
		encoded[i] = clampUnit(a.value[i] / scale)
	}
	return encoded
}

// Control decodes a normalized slice into the current value.
func (a *SliceAdapter) Control(values []float64) error {
	if len(values) < len(a.scales) {
		return fmt.Errorf("adapter: slice %q expects %d values, got %d", a.name, len(a.scales), len(values))
	}
	for i, scale := range a.scales {
		a.value[i] = clampUnit(values[i]) * scale
	}
	return nil
}
