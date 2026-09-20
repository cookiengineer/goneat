package backprop

import (
	"encoding/json"
	"errors"
	"math"

	"github.com/cookiengineer/goneat/engine/neat"
)

// Trainer applies gradients produced by a Graph to a genome's connection
// weights, using a persistent Optimizer so momentum and adaptive state
// survive across calls.
type Trainer struct {
	optimizer   Optimizer
	weightRange float64
}

// NewTrainer creates a trainer. weightRange clips weights to
// [-weightRange, weightRange]; a value <= 0 disables clipping.
func NewTrainer(optimizer Optimizer, weightRange float64) *Trainer {
	return &Trainer{optimizer: optimizer, weightRange: weightRange}
}

// Optimizer returns the underlying optimizer.
func (t *Trainer) Optimizer() Optimizer { return t.optimizer }

// Learn performs a single forward/backward pass and updates the genome
// weights. If network is nil it is built from the genome. The loss value is
// returned.
func (t *Trainer) Learn(genome *neat.Genome, network *neat.Network, inputs, targets []float64, loss Loss) (float64, error) {
	if network == nil {
		built, err := neat.NewNetwork(genome)
		if err != nil {
			return 0, err
		}
		network = built
	}

	graph := NewGraph(network)
	if _, err := graph.Forward(inputs); err != nil {
		return 0, err
	}
	value, err := graph.Backward(loss, targets)
	if err != nil {
		return 0, err
	}
	t.ApplyGradients(genome, graph.WeightGradients())
	return value, nil
}

// ApplyGradients updates every enabled connection using its gradient (zero if
// none was produced) and clips the result.
func (t *Trainer) ApplyGradients(genome *neat.Genome, gradients map[int]float64) {
	for index := range genome.Connections {
		connection := &genome.Connections[index]
		if !connection.Enabled {
			continue
		}
		gradient := gradients[connection.Innovation]
		updated := t.optimizer.Update(connection.Innovation, connection.Weight, gradient)
		if t.weightRange > 0 {
			if updated > t.weightRange {
				updated = t.weightRange
			} else if updated < -t.weightRange {
				updated = -t.weightRange
			}
		}
		if math.IsNaN(updated) || math.IsInf(updated, 0) {
			continue
		}
		connection.Weight = updated
	}
}

type trainerState struct {
	Optimizer   json.RawMessage `json:"optimizer"`
	Kind        string          `json:"kind"`
	WeightRange float64         `json:"weight_range"`
}

// MarshalJSON serializes the trainer and its optimizer state.
func (t *Trainer) MarshalJSON() ([]byte, error) {
	encoded, err := json.Marshal(t.optimizer)
	if err != nil {
		return nil, err
	}
	return json.Marshal(trainerState{
		Optimizer:   encoded,
		Kind:        t.optimizer.Kind(),
		WeightRange: t.weightRange,
	})
}

// UnmarshalJSON restores a trainer and its optimizer state.
func (t *Trainer) UnmarshalJSON(data []byte) error {
	var state trainerState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	switch state.Kind {
	case "sgd":
		optimizer := NewSGD(0, 0)
		if err := json.Unmarshal(state.Optimizer, optimizer); err != nil {
			return err
		}
		t.optimizer = optimizer
	case "rmsprop":
		optimizer := NewRMSProp(0, 0, 0, 0, 0)
		if err := json.Unmarshal(state.Optimizer, optimizer); err != nil {
			return err
		}
		t.optimizer = optimizer
	default:
		return errors.New("backprop: unknown optimizer kind " + state.Kind)
	}
	t.weightRange = state.WeightRange
	return nil
}

// NewTrainerFromState restores a trainer from JSON.
func NewTrainerFromState(data []byte) (*Trainer, error) {
	trainer := &Trainer{}
	if err := json.Unmarshal(data, trainer); err != nil {
		return nil, err
	}
	return trainer, nil
}

// SetRate updates the learning rate of the underlying optimizer.
func (t *Trainer) SetRate(rate float64) {
	switch optimizer := t.optimizer.(type) {
	case *SGD:
		optimizer.Rate = rate
	case *RMSProp:
		optimizer.Rate = rate
	}
}

// Rate returns the current learning rate, or 0 if unknown.
func (t *Trainer) Rate() float64 {
	switch optimizer := t.optimizer.(type) {
	case *SGD:
		return optimizer.Rate
	case *RMSProp:
		return optimizer.Rate
	}
	return 0
}

// WeightRange returns the weight clipping bound.
func (t *Trainer) WeightRange() float64 { return t.weightRange }
