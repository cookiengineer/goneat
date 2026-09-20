package hyperneat

import "errors"

// GridSubstrateLayout places inputs along the bottom edge (y = -1) and outputs
// along the top edge (y = +1), evenly spaced on x. The bias sits at the
// origin. This is the canonical HyperNEAT layout.
type GridSubstrateLayout struct {
	biasCount   int
	inputCount  int
	hiddenCount int
	outputCount int
}

// NewGridSubstrateLayout creates a grid layout.
func NewGridSubstrateLayout(biasCount, inputCount, hiddenCount, outputCount int) *GridSubstrateLayout {
	return &GridSubstrateLayout{
		biasCount:   biasCount,
		inputCount:  inputCount,
		hiddenCount: hiddenCount,
		outputCount: outputCount,
	}
}

// NodePosition implements SubstrateLayout.
func (g *GridSubstrateLayout) NodePosition(index int, nodeType NodeType) (Point, error) {
	if index < 0 {
		return Point{}, errors.New("hyperneat: negative node index")
	}
	switch nodeType {
	case NodeBias:
		if index >= g.biasCount {
			return Point{}, errors.New("hyperneat: bias index out of range")
		}
		return Point{X: 0, Y: 0, Z: 0}, nil
	case NodeInput:
		if index >= g.inputCount {
			return Point{}, errors.New("hyperneat: input index out of range")
		}
		return Point{X: spread(index, g.inputCount), Y: -1, Z: 0}, nil
	case NodeHidden:
		if index >= g.hiddenCount {
			return Point{}, errors.New("hyperneat: hidden index out of range")
		}
		return Point{X: spread(index, g.hiddenCount), Y: 0, Z: 0}, nil
	case NodeOutput:
		if index >= g.outputCount {
			return Point{}, errors.New("hyperneat: output index out of range")
		}
		return Point{X: spread(index, g.outputCount), Y: 1, Z: 0}, nil
	default:
		return Point{}, errors.New("hyperneat: unknown node type")
	}
}

// BiasCount implements SubstrateLayout.
func (g *GridSubstrateLayout) BiasCount() int { return g.biasCount }

// InputCount implements SubstrateLayout.
func (g *GridSubstrateLayout) InputCount() int { return g.inputCount }

// HiddenCount implements SubstrateLayout.
func (g *GridSubstrateLayout) HiddenCount() int { return g.hiddenCount }

// OutputCount implements SubstrateLayout.
func (g *GridSubstrateLayout) OutputCount() int { return g.outputCount }

// spread maps index in [0, count) evenly onto [-1, 1].
func spread(index, count int) float64 {
	if count <= 1 {
		return 0
	}
	return -1 + 2*float64(index)/float64(count-1)
}
