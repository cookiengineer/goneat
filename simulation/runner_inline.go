package simulation

import (
	"context"

	"github.com/cookiengineer/goneat/agent"
)

// InlineRunner executes the lock-step loop sequentially. It is the
// deterministic reference used by tests; ChannelRunner must match it exactly.
type InlineRunner struct{}

// Name implements Runner.
func (InlineRunner) Name() string { return "inline" }

// Run implements Runner.
func (InlineRunner) Run(ctx context.Context, world World, agents []*agent.Agent, config RunConfig) error {
	if err := validate(world, agents); err != nil {
		return err
	}

	outputs := make([][]float64, len(agents))
	for index := range outputs {
		outputs[index] = make([]float64, world.ControlDimensions())
	}

	for tick := 0; tick < config.Ticks; tick++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		sensors := world.Sensors()
		for index, a := range agents {
			if index >= len(sensors) {
				break
			}
			if err := a.Activate(sensors[index], outputs[index]); err != nil {
				return err
			}
		}

		world.Apply(outputs)
		world.Step(config.DT)

		if err := applyFeedback(buildStepContext(tick, world, sensors, outputs, agents), agents, config); err != nil {
			return err
		}

		if allDone(world.Done()) {
			return nil
		}
	}
	return nil
}
