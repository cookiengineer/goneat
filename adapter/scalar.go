package adapter

import (
	"fmt"
	"math"
)

// SquashMode selects how a ScalarAdapter maps an unbounded value into [-1, 1].
type SquashMode int

const (
	// ClampMode maps a bounded value linearly and clips outside the bounds.
	ClampMode SquashMode = iota
	// TanhMode maps an unbounded value through tanh(value/scale).
	TanhMode
)

// ScalarAdapter vectorizes a single floating point value. The adapter owns the
// current value.
type ScalarAdapter struct {
	name         string
	mode         SquashMode
	minimum      float64
	maximum      float64
	scale        float64
	value        float64
	placement    Point
	hasPlacement bool
}

// NewClampedScalar creates a scalar adapter that linearly maps
// [minimum, maximum] onto [-1, 1] and clamps outside it.
func NewClampedScalar(name string, minimum, maximum float64) *ScalarAdapter {
	return &ScalarAdapter{
		name:    name,
		mode:    ClampMode,
		minimum: minimum,
		maximum: maximum,
	}
}

// NewSquashedScalar creates a scalar adapter using tanh with the given scale.
// A scale of 0 defaults to 1.
func NewSquashedScalar(name string, scale float64) *ScalarAdapter {
	if scale == 0 {
		scale = 1
	}
	return &ScalarAdapter{
		name:  name,
		mode:  TanhMode,
		scale: scale,
	}
}

// WithPlacement assigns a substrate placement coordinate.
func (s *ScalarAdapter) WithPlacement(point Point) *ScalarAdapter {
	s.placement = point
	s.hasPlacement = true
	return s
}

// Name returns the adapter name.
func (s *ScalarAdapter) Name() string { return s.name }

// Placement implements Placed.
func (s *ScalarAdapter) Placement() (Point, bool) { return s.placement, s.hasPlacement }

// Dimensions is always 1.
func (s *ScalarAdapter) Dimensions() int { return 1 }

// SetValue sets the current value.
func (s *ScalarAdapter) SetValue(value float64) { s.value = value }

// Value returns the current value.
func (s *ScalarAdapter) Value() float64 { return s.value }

// Sense encodes the current value.
func (s *ScalarAdapter) Sense() []float64 {
	if s.mode == TanhMode {
		return []float64{math.Tanh(s.value / s.scale)}
	}
	return []float64{normalize(s.value, s.minimum, s.maximum)}
}

// Control decodes a normalized value. ClampMode clamps to the configured
// bounds; TanhMode applies atanh with a safety margin.
func (s *ScalarAdapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: scalar %q expects 1 value, got %d", s.name, len(values))
	}
	if s.mode == TanhMode {
		squashed := clampUnit(values[0])
		const limit = 0.999999
		if squashed > limit {
			squashed = limit
		} else if squashed < -limit {
			squashed = -limit
		}
		s.value = math.Atanh(squashed) * s.scale
		return nil
	}
	s.value = denormalize(values[0], s.minimum, s.maximum)
	return nil
}
