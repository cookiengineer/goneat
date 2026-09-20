// Package benchmarks provides standard, self-contained tasks used to compare
// search strategies (in particular the phased-searching modes) with more
// complex input/output decisions than XOR:
//
//   - N-bit parity (combinatorial)
//   - N-multiplexer (combinatorial, a classic NEAT/HyperNEAT benchmark)
//   - CartPole balancing (sequential control, the OpenAI Gym environment)
//
// Each task exposes a fitness function, a solved predicate and an optional
// supervised gradient trainer so the Backprop and Hybrid phases have something
// to learn from.
package benchmarks

import (
	"math"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
)

// Sample is one supervised input/target pair in the normalized world.
type Sample struct {
	Inputs  []float64
	Targets []float64
}

// Task is a benchmark environment.
type Task interface {
	Name() string
	InputCount() int
	OutputCount() int
	// Evaluate returns the fitness of a genome (higher is better).
	Evaluate(genome *neat.Genome) (float64, error)
	// Solved reports whether a genome provably solves the task.
	Solved(genome *neat.Genome) bool
	// Trainer returns the gradient trainer for trainable phases, or nil.
	Trainer() neat.PhaseTrainer
}

// evaluatePatterns runs a network over patterns and returns the mean squared
// error and the classification accuracy at a 0.5 threshold.
func evaluatePatterns(network *neat.Network, patterns []Sample) (float64, float64, error) {
	outputs := make([]float64, network.OutputCount())
	errorSum := 0.0
	correct := 0
	for _, sample := range patterns {
		if err := network.Activate(sample.Inputs, outputs); err != nil {
			return math.Inf(1), 0, err
		}
		for i := range outputs {
			difference := outputs[i] - sample.Targets[i]
			errorSum += difference * difference
			if (outputs[i] >= 0.5) == (sample.Targets[i] >= 0.5) {
				correct++
			}
		}
	}
	total := float64(len(patterns) * len(outputs))
	return errorSum / total, float64(correct) / total, nil
}

// SupervisedTrainer trains a genome on a fixed sample set with backprop. A
// fresh optimizer is created per Train call so genomes do not share state.
type SupervisedTrainer struct {
	Loss        backprop.Loss
	Optimizer   func() backprop.Optimizer
	WeightRange float64
	Epochs      int
	Samples     []Sample
}

// Train implements neat.PhaseTrainer.
func (t *SupervisedTrainer) Train(genome *neat.Genome, iterations int) error {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return err
	}
	optimizer := t.Optimizer
	if optimizer == nil {
		optimizer = func() backprop.Optimizer { return backprop.NewSGD(0.05, 0.9) }
	}
	loss := t.Loss
	if loss == nil {
		loss = backprop.MSELoss{}
	}
	trainer := backprop.NewTrainer(optimizer(), t.WeightRange)
	for i := 0; i < iterations; i++ {
		for _, sample := range t.Samples {
			if _, err := trainer.Learn(genome, network, sample.Inputs, sample.Targets, loss); err != nil {
				return err
			}
		}
	}
	return nil
}
