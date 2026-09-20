package agent

import (
	"math"

	"github.com/cookiengineer/goneat/engine/backprop"
)

// Agent is a competitor in a simulation. It is deliberately brain-agnostic:
// any Brain implementation can drive it, and only brains that also implement
// Trainer receive gradient updates from Reward/Punish.
type Agent struct {
	ID      string
	Alive   bool
	Fitness float64
	Brain   Brain
}

// NewAgent creates a living agent with the given brain.
func NewAgent(id string, brain Brain) *Agent {
	return &Agent{ID: id, Alive: true, Brain: brain}
}

// Activate forwards inputs to the brain.
func (a *Agent) Activate(inputs, outputs []float64) error {
	if a.Brain == nil {
		return nil
	}
	return a.Brain.Activate(inputs, outputs)
}

// Reward adds delta to fitness and, when a training sample is supplied and the
// brain can learn, performs immediate gradient training.
func (a *Agent) Reward(delta float64, training *Training) error {
	a.Fitness += delta
	return a.applyTraining(training)
}

// Punish subtracts the magnitude of delta from fitness and optionally trains.
func (a *Agent) Punish(delta float64, training *Training) error {
	a.Fitness -= math.Abs(delta)
	return a.applyTraining(training)
}

func (a *Agent) applyTraining(training *Training) error {
	if training == nil || a.Brain == nil {
		return nil
	}
	trainer, ok := a.Brain.(Trainer)
	if !ok {
		return nil
	}

	loss := training.Loss
	if loss == nil {
		loss = backprop.MSELoss{}
	}
	iterations := training.Iterations
	if iterations < 1 {
		iterations = 1
	}
	return trainer.Learn(training.Inputs, training.Targets, loss, training.Rate, iterations)
}

// Reset clears per-episode state.
func (a *Agent) Reset() {
	a.Alive = true
	a.Fitness = 0
}
