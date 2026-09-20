// Package adapter translates arbitrary program state into the normalized
// "NEAT world" and back.
//
// Every value a neural network observes is expressed as one or more float64
// values in [-1, 1]. Every value a network controls is written back as one or
// more float64 values in [-1, 1]. Adapters implement either Sensor (read
// side), Control (write side), or both.
package adapter

// Point is a coordinate in the unit hypercube used by HyperNEAT substrates.
// Components are expected to lie in [-1, 1].
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

// Sensor produces a normalized vector describing some part of the controlled
// program. Sense must return exactly Dimensions values in [-1, 1].
type Sensor interface {
	Name() string
	Dimensions() int
	Sense() []float64
}

// Control consumes a normalized vector and applies it to the controlled
// program. It receives exactly Dimensions values in [-1, 1].
type Control interface {
	Name() string
	Dimensions() int
	Control(values []float64) error
}

// Placed is implemented by adapters that know where they belong in a
// HyperNEAT substrate. Placement returns a point in [-1, 1]^3 and whether a
// placement was configured.
type Placed interface {
	Placement() (Point, bool)
}

// clamp normalizes to the [-1, 1] range.
func clampUnit(value float64) float64 {
	if value < -1 {
		return -1
	}
	if value > 1 {
		return 1
	}
	return value
}

// normalize maps a value inside [minimum, maximum] to [-1, 1].
func normalize(value, minimum, maximum float64) float64 {
	if maximum <= minimum {
		return 0
	}
	return 2*(value-minimum)/(maximum-minimum) - 1
}

// denormalize maps a value in [-1, 1] back into [minimum, maximum].
func denormalize(value, minimum, maximum float64) float64 {
	return minimum + (clampUnit(value)+1)/2*(maximum-minimum)
}
