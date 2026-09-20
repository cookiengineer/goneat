package hyperneat

import (
	"errors"

	"github.com/cookiengineer/goneat/engine/neat"
)

// CPPNInputs is the number of coordinates a CPPN consumes: the source and
// target positions in the substrate hypercube.
const CPPNInputs = 6

// CPPN is a Compositional Pattern Producing Network: a neat network that maps
// substrate coordinates to connection weights (and optionally a link
// expression output).
type CPPN struct {
	genome  *neat.Genome
	network *neat.Network
	inputs  []float64
	outputs []float64
}

// NewCPPN builds a CPPN around a genome. The genome must have CPPNInputs
// inputs and at least one output.
func NewCPPN(genome *neat.Genome) (*CPPN, error) {
	if genome == nil {
		return nil, errors.New("hyperneat: nil cppn genome")
	}
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return nil, err
	}
	if network.InputCount() != CPPNInputs {
		return nil, errors.New("hyperneat: cppn must have 6 inputs")
	}
	if network.OutputCount() < 1 {
		return nil, errors.New("hyperneat: cppn must have at least one output")
	}
	return &CPPN{
		genome:  genome,
		network: network,
		inputs:  make([]float64, CPPNInputs),
		outputs: make([]float64, network.OutputCount()),
	}, nil
}

// Genome returns the underlying CPPN genome.
func (c *CPPN) Genome() *neat.Genome { return c.genome }

// OutputCount returns the number of CPPN outputs.
func (c *CPPN) OutputCount() int { return c.network.OutputCount() }

// Query evaluates the CPPN for source and target coordinates. The returned
// slice is owned by the CPPN and is valid until the next Query.
func (c *CPPN) Query(source, target Point) ([]float64, error) {
	c.inputs[0] = source.X
	c.inputs[1] = source.Y
	c.inputs[2] = source.Z
	c.inputs[3] = target.X
	c.inputs[4] = target.Y
	c.inputs[5] = target.Z
	if err := c.network.Activate(c.inputs, c.outputs); err != nil {
		return nil, err
	}
	return c.outputs, nil
}
