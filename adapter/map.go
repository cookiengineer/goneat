package adapter

import "fmt"

// MapAdapter vectorizes a map[string]float64 over a fixed, ordered key set.
// Each key becomes one normalized dimension, which is useful for exposing
// named scalar channels (statistics, sensors by name) to a network. Missing
// keys read as zero. The adapter owns the current map.
type MapAdapter struct {
	name         string
	keys         []string
	minimum      float64
	maximum      float64
	value        map[string]float64
	placement    Point
	hasPlacement bool
}

// NewMap creates a map adapter for the given keys and value range.
func NewMap(name string, keys []string, minimum, maximum float64) *MapAdapter {
	return &MapAdapter{
		name:    name,
		keys:    append([]string(nil), keys...),
		minimum: minimum,
		maximum: maximum,
		value:   make(map[string]float64, len(keys)),
	}
}

// WithPlacement assigns a substrate placement coordinate.
func (a *MapAdapter) WithPlacement(point Point) *MapAdapter {
	a.placement = point
	a.hasPlacement = true
	return a
}

// Name returns the adapter name.
func (a *MapAdapter) Name() string { return a.name }

// Placement implements Placed.
func (a *MapAdapter) Placement() (Point, bool) { return a.placement, a.hasPlacement }

// Dimensions returns the number of keys.
func (a *MapAdapter) Dimensions() int { return len(a.keys) }

// Keys returns the ordered key set.
func (a *MapAdapter) Keys() []string { return append([]string(nil), a.keys...) }

// SetValues replaces the current map. Keys not present in the adapter are
// ignored.
func (a *MapAdapter) SetValues(values map[string]float64) {
	clear(a.value)
	for _, key := range a.keys {
		if value, ok := values[key]; ok {
			a.value[key] = value
		}
	}
}

// Values returns a copy of the current map.
func (a *MapAdapter) Values() map[string]float64 {
	copied := make(map[string]float64, len(a.value))
	for key, value := range a.value {
		copied[key] = value
	}
	return copied
}

// Sense encodes the map in key order.
func (a *MapAdapter) Sense() []float64 {
	encoded := make([]float64, len(a.keys))
	for i, key := range a.keys {
		encoded[i] = normalize(a.value[key], a.minimum, a.maximum)
	}
	return encoded
}

// Control decodes a normalized frame into the current map.
func (a *MapAdapter) Control(values []float64) error {
	if len(values) < len(a.keys) {
		return fmt.Errorf("adapter: map %q expects %d values, got %d", a.name, len(a.keys), len(values))
	}
	clear(a.value)
	for i, key := range a.keys {
		a.value[key] = denormalize(values[i], a.minimum, a.maximum)
	}
	return nil
}
