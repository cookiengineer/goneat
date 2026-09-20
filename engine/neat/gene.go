package neat

// NodeType classifies a node in a genome.
type NodeType uint8

const (
	// NodeInput receives data from the outside world.
	NodeInput NodeType = iota
	// NodeBias always evaluates to 1 and provides a trainable offset.
	NodeBias
	// NodeHidden is an evolved internal node.
	NodeHidden
	// NodeOutput produces a control value for the outside world.
	NodeOutput
)

// String returns a human-readable node type name.
func (t NodeType) String() string {
	switch t {
	case NodeInput:
		return "input"
	case NodeBias:
		return "bias"
	case NodeHidden:
		return "hidden"
	case NodeOutput:
		return "output"
	default:
		return "unknown"
	}
}

// MarshalText implements encoding.TextMarshaler.
func (t NodeType) MarshalText() ([]byte, error) {
	return []byte(t.String()), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (t *NodeType) UnmarshalText(text []byte) error {
	switch string(text) {
	case "input":
		*t = NodeInput
	case "bias":
		*t = NodeBias
	case "hidden":
		*t = NodeHidden
	case "output":
		*t = NodeOutput
	default:
		return &UnknownNodeTypeError{Name: string(text)}
	}
	return nil
}

// UnknownNodeTypeError is returned when decoding an unrecognized node type.
type UnknownNodeTypeError struct {
	Name string
}

func (e *UnknownNodeTypeError) Error() string {
	return "neat: unknown node type " + e.Name
}

// NodeGene describes a node in a genome.
type NodeGene struct {
	ID         int            `json:"id"`
	Type       NodeType       `json:"type"`
	Activation ActivationType `json:"activation"`
}

// ConnectionGene describes a weighted, optionally enabled connection. The
// innovation number uniquely identifies the connection across a population
// and is what makes crossover align genes correctly.
type ConnectionGene struct {
	Innovation int     `json:"innovation"`
	In         int     `json:"in"`
	Out        int     `json:"out"`
	Weight     float64 `json:"weight"`
	Enabled    bool    `json:"enabled"`
}
