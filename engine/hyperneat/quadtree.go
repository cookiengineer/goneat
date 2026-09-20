package hyperneat

import "math"

// ESEvolvableOptions configures ES-HyperNEAT hidden-node discovery.
type ESEvolvableOptions struct {
	// InitialDepth is the initial quadtree resolution.
	InitialDepth int
	// MaximalDepth bounds subdivision.
	MaximalDepth int
	// DivisionThreshold subdivides a region whose CPPN variance exceeds it.
	DivisionThreshold float64
	// VarianceThreshold accepts a leaf as a hidden neuron when its variance is
	// at or below it.
	VarianceThreshold float64
	// BandingThreshold rejects leaves whose local spread exceeds it, avoiding
	// points on sharp boundaries.
	BandingThreshold float64
	// Width and Height define the sampled hypercube (typically 2.0 to cover
	// [-1, 1]).
	Width  float64
	Height float64
}

// DefaultESOptions returns sensible ES-HyperNEAT discovery defaults.
func DefaultESOptions() ESEvolvableOptions {
	return ESEvolvableOptions{
		InitialDepth:      2,
		MaximalDepth:      4,
		DivisionThreshold: 0.1,
		VarianceThreshold: 0.1,
		BandingThreshold:  0.6,
		Width:             2.0,
		Height:            2.0,
	}
}

type quadNode struct {
	x, y   float64
	size   float64
	weight float64
	child  []*quadNode
}

func (n *quadNode) evaluate(cppn *CPPN) error {
	outputs, err := cppn.Query(Point{}, Point{X: n.x, Y: n.y, Z: 0})
	if err != nil {
		return err
	}
	if len(outputs) > 0 {
		n.weight = outputs[0]
	}
	return nil
}

func (n *quadNode) subdivide() {
	half := n.size / 2
	quarter := n.size / 4
	n.child = []*quadNode{
		{x: n.x - quarter, y: n.y - quarter, size: half},
		{x: n.x + quarter, y: n.y - quarter, size: half},
		{x: n.x - quarter, y: n.y + quarter, size: half},
		{x: n.x + quarter, y: n.y + quarter, size: half},
	}
}

// DiscoverHiddenPoints runs a variance-driven quadtree over the hypercube and
// returns the centers of stable regions as candidate hidden neurons.
func DiscoverHiddenPoints(cppn *CPPN, options ESEvolvableOptions) ([]Point, error) {
	if cppn == nil {
		return nil, nil
	}
	if options.Width <= 0 {
		options.Width = 2
	}
	if options.Height <= 0 {
		options.Height = options.Width
	}
	if options.InitialDepth < 1 {
		options.InitialDepth = 1
	}
	if options.MaximalDepth < options.InitialDepth {
		options.MaximalDepth = options.InitialDepth
	}

	root := &quadNode{x: 0, y: 0, size: math.Max(options.Width, options.Height)}
	if err := root.evaluate(cppn); err != nil {
		return nil, err
	}
	if err := expandQuadtree(root, 1, options, cppn); err != nil {
		return nil, err
	}

	var points []Point
	collectHidden(root, options, &points)
	return points, nil
}

func expandQuadtree(node *quadNode, depth int, options ESEvolvableOptions, cppn *CPPN) error {
	node.subdivide()
	for _, child := range node.child {
		if err := child.evaluate(cppn); err != nil {
			return err
		}
	}
	if depth < options.InitialDepth {
		for _, child := range node.child {
			if err := expandQuadtree(child, depth+1, options, cppn); err != nil {
				return err
			}
		}
		return nil
	}
	if variance(node.child) >= options.DivisionThreshold && depth < options.MaximalDepth {
		for _, child := range node.child {
			if err := expandQuadtree(child, depth+1, options, cppn); err != nil {
				return err
			}
		}
	}
	return nil
}

func collectHidden(node *quadNode, options ESEvolvableOptions, points *[]Point) {
	if len(node.child) == 0 {
		if variance(node.child) <= options.VarianceThreshold && nodeSpread(node.child) <= options.BandingThreshold {
			*points = append(*points, Point{X: node.x, Y: node.y, Z: 0})
		}
		return
	}
	if variance(node.child) <= options.VarianceThreshold {
		if nodeSpread(node.child) <= options.BandingThreshold {
			*points = append(*points, Point{X: node.x, Y: node.y, Z: 0})
		}
		return
	}
	for _, child := range node.child {
		collectHidden(child, options, points)
	}
}

func variance(nodes []*quadNode) float64 {
	if len(nodes) < 2 {
		return 0
	}
	mean := 0.0
	for _, node := range nodes {
		mean += node.weight
	}
	mean /= float64(len(nodes))
	accumulator := 0.0
	for _, node := range nodes {
		difference := node.weight - mean
		accumulator += difference * difference
	}
	return accumulator / float64(len(nodes))
}

func nodeSpread(nodes []*quadNode) float64 {
	if len(nodes) == 0 {
		return 0
	}
	minimum, maximum := nodes[0].weight, nodes[0].weight
	for _, node := range nodes[1:] {
		if node.weight < minimum {
			minimum = node.weight
		}
		if node.weight > maximum {
			maximum = node.weight
		}
	}
	return maximum - minimum
}
