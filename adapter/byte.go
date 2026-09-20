package adapter

import (
	"fmt"
	"math"
	"unicode"
)

// ByteAdapter vectorizes a byte (uint8) across the full 0..255 range.
type ByteAdapter struct {
	name         string
	value        byte
	placement    Point
	hasPlacement bool
}

// NewByte creates a byte adapter.
func NewByte(name string) *ByteAdapter { return &ByteAdapter{name: name} }

// WithPlacement assigns a substrate placement coordinate.
func (a *ByteAdapter) WithPlacement(point Point) *ByteAdapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *ByteAdapter) Name() string { return a.name }

// Placement implements Placed.
func (a *ByteAdapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions is always 1.
func (a *ByteAdapter) Dimensions() int { return 1 }

// SetValue sets the current byte.
func (a *ByteAdapter) SetValue(value byte) { a.value = value }

// Value returns the current byte.
func (a *ByteAdapter) Value() byte { return a.value }

// Sense encodes the byte into [-1, 1].
func (a *ByteAdapter) Sense() []float64 {
	return []float64{normalize(float64(a.value), 0, 255)}
}

// Control decodes a normalized value into a byte.
func (a *ByteAdapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: byte %q expects 1 value, got %d", a.name, len(values))
	}
	decoded := math.Round(denormalize(values[0], 0, 255))
	if decoded < 0 {
		decoded = 0
	}
	if decoded > 255 {
		decoded = 255
	}
	a.value = byte(decoded)
	return nil
}

// RuneAdapter vectorizes a Unicode code point within a configurable range.
// The default range covers the full Unicode space; a narrower range such as
// printable ASCII gives finer resolution.
type RuneAdapter struct {
	name         string
	minimum      rune
	maximum      rune
	value        rune
	placement    Point
	hasPlacement bool
}

// NewRune creates a rune adapter covering all of Unicode.
func NewRune(name string) *RuneAdapter {
	return NewRuneRange(name, 0, unicode.MaxRune)
}

// NewRuneRange creates a rune adapter for the inclusive range [minimum,
// maximum].
func NewRuneRange(name string, minimum, maximum rune) *RuneAdapter {
	if maximum < minimum {
		minimum, maximum = maximum, minimum
	}
	return &RuneAdapter{name: name, minimum: minimum, maximum: maximum}
}

// WithPlacement assigns a substrate placement coordinate.
func (a *RuneAdapter) WithPlacement(point Point) *RuneAdapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *RuneAdapter) Name() string { return a.name }

// Placement implements Placed.
func (a *RuneAdapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions is always 1.
func (a *RuneAdapter) Dimensions() int { return 1 }

// SetValue sets the current rune, clamped to the configured range.
func (a *RuneAdapter) SetValue(value rune) {
	if value < a.minimum {
		value = a.minimum
	}
	if value > a.maximum {
		value = a.maximum
	}
	a.value = value
}

// Value returns the current rune.
func (a *RuneAdapter) Value() rune { return a.value }

// Sense encodes the rune.
func (a *RuneAdapter) Sense() []float64 {
	return []float64{normalize(float64(a.value), float64(a.minimum), float64(a.maximum))}
}

// Control decodes a normalized value into a rune.
func (a *RuneAdapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: rune %q expects 1 value, got %d", a.name, len(values))
	}
	decoded := math.Round(denormalize(values[0], float64(a.minimum), float64(a.maximum)))
	if decoded < float64(a.minimum) {
		decoded = float64(a.minimum)
	}
	if decoded > float64(a.maximum) {
		decoded = float64(a.maximum)
	}
	a.value = rune(decoded)
	return nil
}
