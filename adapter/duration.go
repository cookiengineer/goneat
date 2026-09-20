package adapter

import (
	"fmt"
	"math"
	"time"
)

// DurationAdapter vectorizes a time.Duration into [-1, 1] using a fixed range.
// The adapter owns the current value.
type DurationAdapter struct {
	name         string
	minimum      time.Duration
	maximum      time.Duration
	value        time.Duration
	placement    Point
	hasPlacement bool
}

// NewDuration creates a duration adapter for the inclusive range [minimum,
// maximum].
func NewDuration(name string, minimum, maximum time.Duration) *DurationAdapter {
	if maximum < minimum {
		minimum, maximum = maximum, minimum
	}
	return &DurationAdapter{name: name, minimum: minimum, maximum: maximum}
}

// WithPlacement assigns a substrate placement coordinate.
func (a *DurationAdapter) WithPlacement(point Point) *DurationAdapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *DurationAdapter) Name() string { return a.name }

// Placement implements Placed.
func (a *DurationAdapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions is always 1.
func (a *DurationAdapter) Dimensions() int { return 1 }

// SetValue sets the current duration, clamped to the configured range.
func (a *DurationAdapter) SetValue(value time.Duration) {
	if value < a.minimum {
		value = a.minimum
	}
	if value > a.maximum {
		value = a.maximum
	}
	a.value = value
}

// Value returns the current duration.
func (a *DurationAdapter) Value() time.Duration { return a.value }

// Sense encodes the duration.
func (a *DurationAdapter) Sense() []float64 {
	return []float64{normalize(float64(a.value), float64(a.minimum), float64(a.maximum))}
}

// Control decodes a normalized value into a duration, rounded to the nearest
// nanosecond.
func (a *DurationAdapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: duration %q expects 1 value, got %d", a.name, len(values))
	}
	decoded := math.Round(denormalize(values[0], float64(a.minimum), float64(a.maximum)))
	if decoded < float64(a.minimum) {
		decoded = float64(a.minimum)
	}
	if decoded > float64(a.maximum) {
		decoded = float64(a.maximum)
	}
	a.value = time.Duration(decoded)
	return nil
}
