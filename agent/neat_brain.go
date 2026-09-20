package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
)

const (
	lossMSE          = "mse"
	lossCrossEntropy = "cross_entropy"
)

// NEATBrain is a Brain backed by an evolved neat.Genome. It is the bridge
// between the genetic algorithm (which owns the genome) and an Agent.
type NEATBrain struct {
	genome  *neat.Genome
	network *neat.Network
	trainer *backprop.Trainer
	loss    backprop.Loss
}

// NewNEATBrain creates a brain around a genome using a default trainer.
func NewNEATBrain(genome *neat.Genome) (*NEATBrain, error) {
	return NewNEATBrainWithTrainer(
		genome,
		backprop.NewTrainer(backprop.NewSGD(0.1, 0.9), 8),
		backprop.MSELoss{},
	)
}

// NewNEATBrainWithTrainer creates a brain with a specific trainer and loss.
func NewNEATBrainWithTrainer(genome *neat.Genome, trainer *backprop.Trainer, loss backprop.Loss) (*NEATBrain, error) {
	if genome == nil {
		return nil, errors.New("agent: nil genome")
	}
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return nil, err
	}
	if trainer == nil {
		trainer = backprop.NewTrainer(backprop.NewSGD(0.1, 0.9), 8)
	}
	if loss == nil {
		loss = backprop.MSELoss{}
	}
	return &NEATBrain{genome: genome, network: network, trainer: trainer, loss: loss}, nil
}

// Genome returns the underlying genome (the source of truth for evolution).
func (b *NEATBrain) Genome() *neat.Genome { return b.genome }

// Network returns the phenotype network.
func (b *NEATBrain) Network() *neat.Network { return b.network }

// Trainer returns the gradient trainer.
func (b *NEATBrain) Trainer() *backprop.Trainer { return b.trainer }

// Activate implements Brain.
func (b *NEATBrain) Activate(inputs, outputs []float64) error {
	return b.network.Activate(inputs, outputs)
}

// Learn implements Trainer, performing up to iterations gradient steps.
func (b *NEATBrain) Learn(inputs, targets []float64, loss backprop.Loss, rate float64, iterations int) error {
	if loss == nil {
		loss = b.loss
	}
	if rate > 0 {
		b.trainer.SetRate(rate)
	}
	if iterations < 1 {
		iterations = 1
	}
	for i := 0; i < iterations; i++ {
		if _, err := b.trainer.Learn(b.genome, b.network, inputs, targets, loss); err != nil {
			return err
		}
	}
	return nil
}

// Weights implements Trainer.
func (b *NEATBrain) Weights() []float64 { return b.genome.WeightVector() }

// SetWeights implements Trainer, assigning enabled weights in innovation order.
func (b *NEATBrain) SetWeights(weights []float64) error {
	ordered := enabledConnectionsByInnovation(b.genome)
	if len(weights) != len(ordered) {
		return fmt.Errorf("agent: expected %d weights, got %d", len(ordered), len(weights))
	}
	for i, index := range ordered {
		b.genome.Connections[index].Weight = weights[i]
	}
	return nil
}

// Clone implements Brain. It round-trips through JSON so the clone is fully
// independent (genome, network and optimizer state).
func (b *NEATBrain) Clone() Brain {
	data, err := b.Encode()
	if err != nil {
		return nil
	}
	clone := &NEATBrain{}
	if err := clone.Decode(data); err != nil {
		return nil
	}
	return clone
}

type neatBrainState struct {
	Genome  *neat.Genome    `json:"genome"`
	Trainer json.RawMessage `json:"trainer"`
	Loss    string          `json:"loss"`
}

// Encode implements Brain.
func (b *NEATBrain) Encode() ([]byte, error) {
	trainer, err := json.Marshal(b.trainer)
	if err != nil {
		return nil, err
	}
	return json.Marshal(neatBrainState{
		Genome:  b.genome,
		Trainer: trainer,
		Loss:    lossName(b.loss),
	})
}

// Decode implements Brain.
func (b *NEATBrain) Decode(data []byte) error {
	var state neatBrainState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	trainer, err := backprop.NewTrainerFromState(state.Trainer)
	if err != nil {
		return err
	}
	loss, err := lossFromName(state.Loss)
	if err != nil {
		return err
	}
	b.genome = state.Genome
	b.trainer = trainer
	b.loss = loss
	network, err := neat.NewNetwork(b.genome)
	if err != nil {
		return err
	}
	b.network = network
	return nil
}

func lossName(loss backprop.Loss) string {
	switch loss.(type) {
	case backprop.CrossEntropyLoss:
		return lossCrossEntropy
	default:
		return lossMSE
	}
}

func lossFromName(name string) (backprop.Loss, error) {
	switch name {
	case "", lossMSE:
		return backprop.MSELoss{}, nil
	case lossCrossEntropy:
		return backprop.CrossEntropyLoss{}, nil
	default:
		return nil, fmt.Errorf("agent: unknown loss %q", name)
	}
}

func enabledConnectionsByInnovation(genome *neat.Genome) []int {
	indices := make([]int, 0, len(genome.Connections))
	for index, connection := range genome.Connections {
		if connection.Enabled {
			indices = append(indices, index)
		}
	}
	sort.Slice(indices, func(a, b int) bool {
		return genome.Connections[indices[a]].Innovation < genome.Connections[indices[b]].Innovation
	})
	return indices
}
