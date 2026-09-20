package adapter

import "fmt"

// BoolAdapter vectorizes a boolean as -1 (false) or +1 (true). The adapter
// owns the current value.
type BoolAdapter struct {
	name         string
	value        bool
	placement    Point
	hasPlacement bool
}

// NewBool creates a boolean adapter.
func NewBool(name string) *BoolAdapter {
	return &BoolAdapter{name: name}
}

// WithPlacement assigns a substrate placement coordinate.
func (b *BoolAdapter) WithPlacement(point Point) *BoolAdapter {
	b.placement = point
	b.hasPlacement = true
	return b
}

// Name returns the adapter name.
func (b *BoolAdapter) Name() string { return b.name }

// Placement implements Placed.
func (b *BoolAdapter) Placement() (Point, bool) { return b.placement, b.hasPlacement }

// Dimensions is always 1.
func (b *BoolAdapter) Dimensions() int { return 1 }

// SetValue sets the current value.
func (b *BoolAdapter) SetValue(value bool) { b.value = value }

// Value returns the current value.
func (b *BoolAdapter) Value() bool { return b.value }

// Sense encodes the current value as -1 or +1.
func (b *BoolAdapter) Sense() []float64 {
	if b.value {
		return []float64{1}
	}
	return []float64{-1}
}

// Control decodes a normalized value, thresholding at 0.
func (b *BoolAdapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: bool %q expects 1 value, got %d", b.name, len(values))
	}
	b.value = values[0] >= 0
	return nil
}
