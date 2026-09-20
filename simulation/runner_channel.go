package simulation

import (
	"context"

	"github.com/cookiengineer/goneat/agent"
)

// ActivationResult carries an agent's control frame or the error that occurred
// while producing it.
type ActivationResult struct {
	Outputs []float64
	Err     error
}

// Port wires one agent goroutine to the world goroutine.
type Port struct {
	Inputs  chan []float64
	Outputs chan ActivationResult
	Done    chan struct{}
}

// ChannelRunner evaluates agents concurrently, one goroutine per agent, with a
// lock-step barrier each tick. It scales to many cores while producing exactly
// the same result as InlineRunner for a given seed.
type ChannelRunner struct {
	// Buffer is the channel buffer size. Zero gives a strict per-tick barrier.
	Buffer int
}

// Name implements Runner.
func (ChannelRunner) Name() string { return "channel" }

// Run implements Runner.
func (r ChannelRunner) Run(ctx context.Context, world World, agents []*agent.Agent, config RunConfig) error {
	if err := validate(world, agents); err != nil {
		return err
	}

	buffer := r.Buffer
	if buffer < 0 {
		buffer = 0
	}
	outputDimensions := world.ControlDimensions()

	ports := make([]Port, len(agents))
	for index := range ports {
		ports[index] = Port{
			Inputs:  make(chan []float64, buffer),
			Outputs: make(chan ActivationResult, buffer),
			Done:    make(chan struct{}),
		}
	}

	for index, current := range agents {
		go func(a *agent.Agent, port Port) {
			defer close(port.Done)
			for frame := range port.Inputs {
				outputs := make([]float64, outputDimensions)
				err := a.Activate(frame, outputs)
				port.Outputs <- ActivationResult{Outputs: outputs, Err: err}
			}
		}(current, ports[index])
	}

	var runErr error
	for tick := 0; tick < config.Ticks; tick++ {
		if err := ctx.Err(); err != nil {
			runErr = err
			break
		}

		sensors := world.Sensors()
		for index := range agents {
			if index < len(sensors) {
				ports[index].Inputs <- sensors[index]
			}
		}

		controls := make([][]float64, len(agents))
		for index := range agents {
			result := <-ports[index].Outputs
			if result.Err != nil {
				runErr = result.Err
			}
			controls[index] = result.Outputs
		}
		if runErr != nil {
			break
		}

		world.Apply(controls)
		world.Step(config.DT)

		if err := applyFeedback(buildStepContext(tick, world, sensors, controls, agents), agents, config); err != nil {
			runErr = err
			break
		}

		if allDone(world.Done()) {
			break
		}
	}

	for index := range ports {
		close(ports[index].Inputs)
	}
	for index := range ports {
		<-ports[index].Done
	}
	return runErr
}
