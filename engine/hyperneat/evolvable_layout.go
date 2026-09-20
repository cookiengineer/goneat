package hyperneat

import "errors"

// MappedEvolvableSubstrateLayout is a substrate whose hidden neurons are
// discovered by ES-HyperNEAT rather than fixed by hand. Inputs and outputs are
// placed on opposite edges; hidden points are added from quadtree discovery.
type MappedEvolvableSubstrateLayout struct {
	inputCount   int
	outputCount  int
	hiddenPoints []Point
}

// NewMappedEvolvableSubstrateLayout creates a layout with no hidden points yet.
func NewMappedEvolvableSubstrateLayout(inputCount, outputCount int) *MappedEvolvableSubstrateLayout {
	return &MappedEvolvableSubstrateLayout{inputCount: inputCount, outputCount: outputCount}
}

// AddHiddenPoint registers a discovered hidden neuron.
func (l *MappedEvolvableSubstrateLayout) AddHiddenPoint(point Point) {
	l.hiddenPoints = append(l.hiddenPoints, point)
}

// SetHiddenPoints replaces the discovered hidden neurons.
func (l *MappedEvolvableSubstrateLayout) SetHiddenPoints(points []Point) {
	l.hiddenPoints = append([]Point(nil), points...)
}

// HiddenPoints returns the discovered hidden neurons.
func (l *MappedEvolvableSubstrateLayout) HiddenPoints() []Point {
	return append([]Point(nil), l.hiddenPoints...)
}

// NodePosition implements SubstrateLayout.
func (l *MappedEvolvableSubstrateLayout) NodePosition(index int, nodeType NodeType) (Point, error) {
	if index < 0 {
		return Point{}, errors.New("hyperneat: negative node index")
	}
	switch nodeType {
	case NodeBias:
		if index != 0 {
			return Point{}, errors.New("hyperneat: only one bias supported")
		}
		return Point{X: 0, Y: 0, Z: 0}, nil
	case NodeInput:
		if index >= l.inputCount {
			return Point{}, errors.New("hyperneat: input index out of range")
		}
		return Point{X: spread(index, l.inputCount), Y: -1, Z: 0}, nil
	case NodeHidden:
		if index >= len(l.hiddenPoints) {
			return Point{}, errors.New("hyperneat: hidden index out of range")
		}
		return l.hiddenPoints[index], nil
	case NodeOutput:
		if index >= l.outputCount {
			return Point{}, errors.New("hyperneat: output index out of range")
		}
		return Point{X: spread(index, l.outputCount), Y: 1, Z: 0}, nil
	default:
		return Point{}, errors.New("hyperneat: unknown node type")
	}
}

// BiasCount implements SubstrateLayout.
func (l *MappedEvolvableSubstrateLayout) BiasCount() int { return 1 }

// InputCount implements SubstrateLayout.
func (l *MappedEvolvableSubstrateLayout) InputCount() int { return l.inputCount }

// HiddenCount implements SubstrateLayout.
func (l *MappedEvolvableSubstrateLayout) HiddenCount() int { return len(l.hiddenPoints) }

// OutputCount implements SubstrateLayout.
func (l *MappedEvolvableSubstrateLayout) OutputCount() int { return l.outputCount }
