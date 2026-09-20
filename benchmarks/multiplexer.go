package benchmarks

import (
	"fmt"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
)

// Multiplexer is the N-multiplexer task: the first addressBits inputs select
// one of the remaining data inputs, and the network must output that bit. The
// 6-multiplexer (2 address bits) and 11-multiplexer (3 address bits) are
// classic NEAT / HyperNEAT benchmarks.
type Multiplexer struct {
	addressBits  int
	dataBits     int
	patterns     []Sample
	evalPatterns []Sample
}

// NewMultiplexer creates an N-multiplexer with the given number of address
// bits.
func NewMultiplexer(addressBits int) *Multiplexer {
	dataBits := 1 << addressBits
	inputs := addressBits + dataBits
	patterns := make([]Sample, 0, dataBits<<dataBits)

	for address := 0; address < dataBits; address++ {
		for data := 0; data < (1 << dataBits); data++ {
			sample := Sample{
				Inputs:  make([]float64, inputs),
				Targets: []float64{float64((data >> address) & 1)},
			}
			for bit := 0; bit < addressBits; bit++ {
				sample.Inputs[bit] = float64((address >> bit) & 1)
			}
			for bit := 0; bit < dataBits; bit++ {
				sample.Inputs[addressBits+bit] = float64((data >> bit) & 1)
			}
			patterns = append(patterns, sample)
		}
	}

	// Keep evaluation affordable for the 11-multiplexer by striding the data
	// space; the full set is still used to declare a genome solved.
	evalPatterns := patterns
	if len(patterns) > 256 {
		stride := len(patterns) / 256
		evalPatterns = make([]Sample, 0, 256)
		for i := 0; i < len(patterns); i += stride {
			evalPatterns = append(evalPatterns, patterns[i])
		}
	}

	return &Multiplexer{
		addressBits:  addressBits,
		dataBits:     dataBits,
		patterns:     patterns,
		evalPatterns: evalPatterns,
	}
}

// Name implements Task.
func (m *Multiplexer) Name() string { return fmt.Sprintf("mux%d", m.addressBits+m.dataBits) }

// InputCount implements Task.
func (m *Multiplexer) InputCount() int { return m.addressBits + m.dataBits }

// OutputCount implements Task.
func (m *Multiplexer) OutputCount() int { return 1 }

// Evaluate implements Task.
func (m *Multiplexer) Evaluate(genome *neat.Genome) (float64, error) {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return 0, err
	}
	meanSquaredError, _, err := evaluatePatterns(network, m.evalPatterns)
	if err != nil {
		return 0, err
	}
	return 1 / (1 + meanSquaredError), nil
}

// Solved implements Task.
func (m *Multiplexer) Solved(genome *neat.Genome) bool {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return false
	}
	_, accuracy, err := evaluatePatterns(network, m.patterns)
	return err == nil && accuracy == 1
}

// Trainer implements Task.
func (m *Multiplexer) Trainer() neat.PhaseTrainer {
	return &SupervisedTrainer{
		Optimizer:   func() backprop.Optimizer { return backprop.NewSGD(0.2, 0.9) },
		WeightRange: 10,
		Samples:     m.evalPatterns,
	}
}
