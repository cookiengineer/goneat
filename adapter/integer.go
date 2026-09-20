package adapter

import (
	"fmt"
	"math"
)

// IntAdapter vectorizes a signed integer in a fixed range onto [-1, 1]. The
// adapter owns the current value.
type IntAdapter struct {
	name         string
	minimum      int64
	maximum      int64
	value        int64
	placement    Point
	hasPlacement bool
}

// NewInt creates a signed integer adapter for the inclusive range [minimum,
// maximum].
func NewInt(name string, minimum, maximum int64) *IntAdapter {
	if maximum < minimum {
		minimum, maximum = maximum, minimum
	}
	return &IntAdapter{name: name, minimum: minimum, maximum: maximum}
}

// WithPlacement assigns a substrate placement coordinate.
func (a *IntAdapter) WithPlacement(point Point) *IntAdapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *IntAdapter) Name() string { return a.name }

// Placement implements Placed.
func (a *IntAdapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions is always 1.
func (a *IntAdapter) Dimensions() int { return 1 }

// SetValue sets the current value, clamped to the configured range.
func (a *IntAdapter) SetValue(value int64) {
	if value < a.minimum {
		value = a.minimum
	}
	if value > a.maximum {
		value = a.maximum
	}
	a.value = value
}

// Value returns the current value.
func (a *IntAdapter) Value() int64 { return a.value }

// Sense encodes the current value.
func (a *IntAdapter) Sense() []float64 {
	return []float64{normalize(float64(a.value), float64(a.minimum), float64(a.maximum))}
}

// Control decodes a normalized value by rounding to the nearest integer.
func (a *IntAdapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: int %q expects 1 value, got %d", a.name, len(values))
	}
	decoded := math.Round(denormalize(values[0], float64(a.minimum), float64(a.maximum)))
	if decoded < float64(a.minimum) {
		decoded = float64(a.minimum)
	}
	if decoded > float64(a.maximum) {
		decoded = float64(a.maximum)
	}
	a.value = int64(decoded)
	return nil
}

// UintAdapter vectorizes an unsigned integer in a fixed range onto [-1, 1].
type UintAdapter struct {
	name         string
	minimum      uint64
	maximum      uint64
	value        uint64
	placement    Point
	hasPlacement bool
}

// NewUint creates an unsigned integer adapter for the inclusive range
// [minimum, maximum].
func NewUint(name string, minimum, maximum uint64) *UintAdapter {
	if maximum < minimum {
		minimum, maximum = maximum, minimum
	}
	return &UintAdapter{name: name, minimum: minimum, maximum: maximum}
}

// WithPlacement assigns a substrate placement coordinate.
func (a *UintAdapter) WithPlacement(point Point) *UintAdapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *UintAdapter) Name() string { return a.name }

// Placement implements Placed.
func (a *UintAdapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions is always 1.
func (a *UintAdapter) Dimensions() int { return 1 }

// SetValue sets the current value, clamped to the configured range.
func (a *UintAdapter) SetValue(value uint64) {
	if value < a.minimum {
		value = a.minimum
	}
	if value > a.maximum {
		value = a.maximum
	}
	a.value = value
}

// Value returns the current value.
func (a *UintAdapter) Value() uint64 { return a.value }

// Sense encodes the current value.
func (a *UintAdapter) Sense() []float64 {
	return []float64{normalize(float64(a.value), float64(a.minimum), float64(a.maximum))}
}

// Control decodes a normalized value by rounding to the nearest integer.
func (a *UintAdapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: uint %q expects 1 value, got %d", a.name, len(values))
	}
	decoded := math.Round(denormalize(values[0], float64(a.minimum), float64(a.maximum)))
	if decoded < float64(a.minimum) {
		decoded = float64(a.minimum)
	}
	if decoded > float64(a.maximum) {
		decoded = float64(a.maximum)
	}
	if decoded < 0 {
		decoded = 0
	}
	a.value = uint64(decoded)
	return nil
}
