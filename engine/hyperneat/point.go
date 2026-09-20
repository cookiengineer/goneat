// Package hyperneat implements HyperNEAT and Evolvable-Substrate HyperNEAT on
// top of the goneat NEAT core.
//
// HyperNEAT uses a Compositional Pattern Producing Network (CPPN) to generate
// the weights of a larger substrate network. A connection weight is a function
// of the coordinates of its source and target neurons in a unit hypercube,
// which lets geometric structure such as mirrored sensors transfer naturally.
package hyperneat

// Point is a coordinate in the substrate hypercube. Components lie in [-1, 1].
type Point struct {
	X float64
	Y float64
	Z float64
}

// MirrorX returns the point reflected across the YZ plane.
func (p Point) MirrorX() Point { return Point{X: -p.X, Y: p.Y, Z: p.Z} }

// MirrorY returns the point reflected across the XZ plane.
func (p Point) MirrorY() Point { return Point{X: p.X, Y: -p.Y, Z: p.Z} }
