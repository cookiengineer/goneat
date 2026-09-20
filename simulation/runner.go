package simulation

import (
	"context"
	"errors"

	"github.com/cookiengineer/goneat/agent"
)

// RunConfig controls a single episode.
type RunConfig struct {
	// Ticks is the maximum number of ticks; the episode may end earlier when
	// every agent reports Done.
	Ticks int
	// DT is the timestep passed to World.Step.
	DT float64
	// Reward and Punish are the user callbacks. Either may be nil.
	Reward RewardFunc
	Punish PunishFunc
}

// Runner drives the lock-step loop between a World and a set of Agents. Two
// implementations are provided: InlineRunner (sequential, deterministic
// reference) and ChannelRunner (one goroutine per agent).
type Runner interface {
	Name() string
	Run(ctx context.Context, world World, agents []*agent.Agent, config RunConfig) error
}

// buildStepContext assembles the per-tick context handed to callbacks.
func buildStepContext(tick int, world World, sensors, controls [][]float64, agents []*agent.Agent) *StepContext {
	ids := make([]string, len(agents))
	for index, a := range agents {
		ids[index] = a.ID
	}
	return &StepContext{
		Tick:     tick,
		World:    world,
		Sensors:  sensors,
		Controls: controls,
		AgentIDs: ids,
	}
}

// applyFeedback invokes the user callbacks for every agent in order.
func applyFeedback(step *StepContext, agents []*agent.Agent, config RunConfig) error {
	for _, a := range agents {
		if config.Reward != nil {
			delta, training := config.Reward(step, a)
			if delta != 0 || training != nil {
				if err := a.Reward(delta, training); err != nil {
					return err
				}
			}
		}
		if config.Punish != nil {
			delta, training := config.Punish(step, a)
			if delta != 0 || training != nil {
				if err := a.Punish(delta, training); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// allDone reports whether every agent has finished.
func allDone(done []bool) bool {
	if len(done) == 0 {
		return true
	}
	for _, finished := range done {
		if !finished {
			return false
		}
	}
	return true
}

func validate(world World, agents []*agent.Agent) error {
	if world == nil {
		return errors.New("simulation: nil world")
	}
	if len(agents) == 0 {
		return errors.New("simulation: no agents")
	}
	if world.SensorDimensions() <= 0 || world.ControlDimensions() <= 0 {
		return errors.New("simulation: world dimensions must be positive")
	}
	return nil
}
