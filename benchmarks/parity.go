package benchmarks

import (
	"fmt"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
)

// Parity is the N-bit parity task: the network must output 1 when the number
// of set input bits is odd, and 0 otherwise. It is a standard NEAT benchmark
// and scales smoothly from trivial (2 bits) to hard (9+ bits).
type Parity struct {
	bits     int
	patterns []Sample
}

// NewParity creates an N-bit parity task.
func NewParity(bits int) *Parity {
	count := 1 << bits
	patterns := make([]Sample, count)
	for value := 0; value < count; value++ {
		inputs := make([]float64, bits)
		ones := 0
		for bit := 0; bit < bits; bit++ {
			set := (value >> bit) & 1
			inputs[bit] = float64(set)
			ones += set
		}
		patterns[value] = Sample{Inputs: inputs, Targets: []float64{float64(ones % 2)}}
	}
	return &Parity{bits: bits, patterns: patterns}
}

// Name implements Task.
func (p *Parity) Name() string { return fmt.Sprintf("parity%d", p.bits) }

// InputCount implements Task.
func (p *Parity) InputCount() int { return p.bits }

// OutputCount implements Task.
func (p *Parity) OutputCount() int { return 1 }

// Evaluate implements Task.
func (p *Parity) Evaluate(genome *neat.Genome) (float64, error) {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return 0, err
	}
	meanSquaredError, _, err := evaluatePatterns(network, p.patterns)
	if err != nil {
		return 0, err
	}
	return 1 / (1 + meanSquaredError), nil
}

// Solved implements Task.
func (p *Parity) Solved(genome *neat.Genome) bool {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return false
	}
	_, accuracy, err := evaluatePatterns(network, p.patterns)
	return err == nil && accuracy == 1
}

// Trainer implements Task.
func (p *Parity) Trainer() neat.PhaseTrainer {
	return &SupervisedTrainer{
		Optimizer:   func() backprop.Optimizer { return backprop.NewSGD(0.2, 0.9) },
		WeightRange: 10,
		Samples:     p.patterns,
	}
}
