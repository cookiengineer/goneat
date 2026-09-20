package adapter

import (
	"fmt"
	"math"
)

// Float32Adapter vectorizes a single float32 value. It mirrors ScalarAdapter
// but stores a native float32, which is convenient when the controlled program
// uses single precision. The adapter owns the current value.
type Float32Adapter struct {
	name         string
	mode         SquashMode
	minimum      float64
	maximum      float64
	scale        float64
	value        float32
	placement    Point
	hasPlacement bool
}

// NewClampedFloat32 creates an adapter that linearly maps [minimum, maximum]
// onto [-1, 1] and clamps outside it.
func NewClampedFloat32(name string, minimum, maximum float64) *Float32Adapter {
	return &Float32Adapter{name: name, mode: ClampMode, minimum: minimum, maximum: maximum}
}

// NewSquashedFloat32 creates an adapter using tanh with the given scale. A
// scale of 0 defaults to 1.
func NewSquashedFloat32(name string, scale float64) *Float32Adapter {
	if scale == 0 {
		scale = 1
	}
	return &Float32Adapter{name: name, mode: TanhMode, scale: scale}
}

// WithPlacement assigns a substrate placement coordinate.
func (a *Float32Adapter) WithPlacement(point Point) *Float32Adapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *Float32Adapter) Name() string { return a.name }

// Placement implements Placed.
func (a *Float32Adapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions is always 1.
func (a *Float32Adapter) Dimensions() int { return 1 }

// SetValue sets the current value.
func (a *Float32Adapter) SetValue(value float32) { a.value = value }

// Value returns the current value.
func (a *Float32Adapter) Value() float32 { return a.value }

// Sense encodes the current value.
func (a *Float32Adapter) Sense() []float64 {
	value := float64(a.value)
	if a.mode == TanhMode {
		return []float64{math.Tanh(value / a.scale)}
	}
	return []float64{normalize(value, a.minimum, a.maximum)}
}

// Control decodes a normalized value. ClampMode clamps to the configured
// bounds; TanhMode applies atanh with a safety margin.
func (a *Float32Adapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: float32 %q expects 1 value, got %d", a.name, len(values))
	}
	if a.mode == TanhMode {
		a.value = float32(atanhSquash(values[0]) * a.scale)
		return nil
	}
	a.value = float32(denormalize(values[0], a.minimum, a.maximum))
	return nil
}

// atanhSquash maps a normalized value through atanh, clamping away from the
// asymptotes so the result stays finite.
func atanhSquash(value float64) float64 {
	const limit = 0.999999
	squashed := clampUnit(value)
	if squashed > limit {
		squashed = limit
	} else if squashed < -limit {
		squashed = -limit
	}
	return math.Atanh(squashed)
}
