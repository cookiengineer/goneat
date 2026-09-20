package hyperneat

import "github.com/cookiengineer/goneat/engine/neat"

// NodeType classifies a substrate neuron.
type NodeType uint8

const (
	// NodeBias is a constant-one input.
	NodeBias NodeType = iota
	// NodeInput receives the task input.
	NodeInput
	// NodeHidden is an evolved or discovered intermediate neuron.
	NodeHidden
	// NodeOutput produces the task output.
	NodeOutput
)

// String returns a human-readable node type.
func (t NodeType) String() string {
	switch t {
	case NodeBias:
		return "bias"
	case NodeInput:
		return "input"
	case NodeHidden:
		return "hidden"
	case NodeOutput:
		return "output"
	default:
		return "unknown"
	}
}

// SubstrateLayout places neurons in the unit hypercube. Implementations define
// the geometry that gives HyperNEAT its generalization properties.
type SubstrateLayout interface {
	// NodePosition returns the coordinate of the index-th neuron of a type.
	NodePosition(index int, nodeType NodeType) (Point, error)
	BiasCount() int
	InputCount() int
	HiddenCount() int
	OutputCount() int
}

// Options configures substrate weight generation.
type Options struct {
	// LinkThreshold is the minimum absolute CPPN output required to express a
	// link when LEO is disabled.
	LinkThreshold float64
	// WeightRange scales generated weights; weights lie in
	// [-WeightRange, +WeightRange].
	WeightRange float64
	// UseLEO enables the Link Expression Output: a second CPPN output greater
	// than zero expresses the link, and the first output is its raw weight.
	UseLEO bool
	// HiddenActivation is the activation of generated hidden neurons.
	HiddenActivation neat.ActivationType
	// OutputActivation is the activation of output neurons.
	OutputActivation neat.ActivationType
}

// DefaultOptions returns sensible substrate defaults.
func DefaultOptions() Options {
	return Options{
		LinkThreshold:    0.2,
		WeightRange:      3.0,
		UseLEO:           false,
		HiddenActivation: neat.ActivationSigmoid,
		OutputActivation: neat.ActivationSigmoid,
	}
}
