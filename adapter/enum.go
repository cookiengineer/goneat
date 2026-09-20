package adapter

import (
	"fmt"
	"math"
)

// EnumAdapter vectorizes a categorical value.
//
// In ordinal mode (default) a category is mapped onto a single dimension:
// the first category becomes -1, the last becomes +1, and intermediate
// categories are evenly spaced. In one-hot mode each category gets its own
// dimension: +1 for the active category and -1 for the rest.
type EnumAdapter struct {
	name         string
	values       []string
	oneHot       bool
	index        int
	placement    Point
	hasPlacement bool
}

// NewEnum creates an enum adapter for the given ordered categories.
func NewEnum(name string, values []string) *EnumAdapter {
	return &EnumAdapter{
		name:   name,
		values: append([]string(nil), values...),
	}
}

// AsOneHot switches the adapter to one-hot encoding.
func (e *EnumAdapter) AsOneHot() *EnumAdapter {
	e.oneHot = true
	return e
}

// WithPlacement assigns a substrate placement coordinate.
func (e *EnumAdapter) WithPlacement(point Point) *EnumAdapter {
	e.placement = point
	e.hasPlacement = true
	return e
}

// Name returns the adapter name.
func (e *EnumAdapter) Name() string { return e.name }

// Values returns the ordered categories.
func (e *EnumAdapter) Values() []string { return append([]string(nil), e.values...) }

// OneHot reports whether one-hot encoding is active.
func (e *EnumAdapter) OneHot() bool { return e.oneHot }

// Placement implements Placed.
func (e *EnumAdapter) Placement() (Point, bool) { return e.placement, e.hasPlacement }

// Dimensions returns 1 for ordinal mode and len(values) for one-hot mode.
func (e *EnumAdapter) Dimensions() int {
	if e.oneHot {
		return len(e.values)
	}
	return 1
}

// SetIndex sets the current category by index.
func (e *EnumAdapter) SetIndex(index int) error {
	if index < 0 || index >= len(e.values) {
		return fmt.Errorf("adapter: enum %q index %d out of range", e.name, index)
	}
	e.index = index
	return nil
}

// SetValue sets the current category by name.
func (e *EnumAdapter) SetValue(value string) error {
	for i, candidate := range e.values {
		if candidate == value {
			e.index = i
			return nil
		}
	}
	return fmt.Errorf("adapter: enum %q has no value %q", e.name, value)
}

// Index returns the current category index.
func (e *EnumAdapter) Index() int { return e.index }

// Value returns the current category name.
func (e *EnumAdapter) Value() string {
	if e.index < 0 || e.index >= len(e.values) {
		return ""
	}
	return e.values[e.index]
}

// Sense encodes the current category.
func (e *EnumAdapter) Sense() []float64 {
	if e.oneHot {
		out := make([]float64, len(e.values))
		for i := range out {
			if i == e.index {
				out[i] = 1
			} else {
				out[i] = -1
			}
		}
		return out
	}
	return []float64{normalizeOrdinal(e.index, len(e.values))}
}

// Control decodes a normalized vector back into the current category.
func (e *EnumAdapter) Control(values []float64) error {
	if e.oneHot {
		if len(values) < len(e.values) {
			return fmt.Errorf("adapter: enum %q expects %d values, got %d", e.name, len(e.values), len(values))
		}
		best, bestValue := 0, math.Inf(-1)
		for i := 0; i < len(e.values); i++ {
			if values[i] > bestValue {
				bestValue = values[i]
				best = i
			}
		}
		e.index = best
		return nil
	}
	if len(values) < 1 {
		return fmt.Errorf("adapter: enum %q expects 1 value, got %d", e.name, len(values))
	}
	e.index = denormalizeOrdinal(values[0], len(e.values))
	return nil
}

func normalizeOrdinal(index, count int) float64 {
	if count <= 1 {
		return 0
	}
	return -1 + 2*float64(index)/float64(count-1)
}

func denormalizeOrdinal(value float64, count int) int {
	if count <= 1 {
		return 0
	}
	index := int(math.Round((clampUnit(value) + 1) / 2 * float64(count-1)))
	if index < 0 {
		return 0
	}
	if index >= count {
		return count - 1
	}
	return index
}
