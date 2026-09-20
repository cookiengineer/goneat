package main

import (
	"github.com/cookiengineer/goneat/agent"
	"github.com/cookiengineer/goneat/simulation"
)

// PongReward is the user-defined reward callback. It measures the lane event
// (hit, miss, score) and translates the ball height into a normalized training
// target for the agent's paddle. This is exactly the "translate it yourself"
// step: the simulation knows nothing about paddles or balls, only normalized
// frames.
func PongReward(step *simulation.StepContext, a *agent.Agent) (float64, *agent.Training) {
	world, ok := step.World.(*PongWorld)
	if !ok {
		return 0, nil
	}
	index := step.AgentIndex(a.ID)
	if index < 0 || index >= len(step.Sensors) {
		return 0, nil
	}

	training := &agent.Training{
		Inputs:     append([]float64(nil), step.Sensors[index]...),
		Targets:    []float64{world.BallY(index)},
		Rate:       0.05,
		Iterations: 1,
	}
	return world.Event(index), training
}

// PongScoreReward reports lane events as fitness but never trains. It is used
// to evaluate a frozen policy fairly (no learning during evaluation).
func PongScoreReward(step *simulation.StepContext, a *agent.Agent) (float64, *agent.Training) {
	world, ok := step.World.(*PongWorld)
	if !ok {
		return 0, nil
	}
	index := step.AgentIndex(a.ID)
	if index < 0 {
		return 0, nil
	}
	return world.Event(index), nil
}
