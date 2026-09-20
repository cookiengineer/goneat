// Package agent defines the brain-agnostic Agent abstraction. An Agent owns a
// fitness value and a Brain, and exposes Reward/Punish entry points that
// adjust fitness and optionally trigger gradient learning on the brain.
package agent

import "github.com/cookiengineer/goneat/engine/backprop"

// Brain is the minimal interface an Agent needs to act. Brains are
// interchangeable: NEAT-evolved networks, fixed backprop networks and
// HyperNEAT substrates all satisfy it.
type Brain interface {
	// Activate maps inputs onto outputs. Both slices are sized by the caller.
	Activate(inputs, outputs []float64) error
	// Clone returns a deep, independent copy.
	Clone() Brain
	// Encode serializes the brain for snapshots.
	Encode() ([]byte, error)
	// Decode restores a brain from Encode's output.
	Decode(data []byte) error
}

// Trainer is implemented by brains that support gradient learning. Reward and
// Punish call Learn when a Training sample is supplied.
type Trainer interface {
	Learn(inputs, targets []float64, loss backprop.Loss, rate float64, iterations int) error
}

// BrainConfig describes the dimensions of a brain to be constructed.
type BrainConfig struct {
	InputDimensions  int
	OutputDimensions int
}

// Training is one supervised sample produced by a user's Reward/Punish
// callback. Inputs and Targets are already expressed in the normalized
// [-1, 1] world, so the user fully controls the translation.
type Training struct {
	Inputs     []float64
	Targets    []float64
	Loss       backprop.Loss
	Rate       float64
	Iterations int
}
