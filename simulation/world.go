// Package simulation wires agents, brains and a controllable world together
// into a lock-step, parallel training loop. It is the layer a user builds
// their own environment on top of.
package simulation

// World is the user's environment. It observes agents through normalized
// sensor frames and steers them through normalized control frames. Both frames
// use values in [-1, 1].
//
// Implementations are free to represent the underlying program however they
// like; the normalized frames are the only contract with the NEAT world.
type World interface {
	// Reset restarts the environment for a new episode.
	Reset(seed int64)
	// SensorDimensions is the size of one agent's sensor frame.
	SensorDimensions() int
	// ControlDimensions is the size of one agent's control frame.
	ControlDimensions() int
	// Sensors returns one frame per agent.
	Sensors() [][]float64
	// Apply delivers one control frame per agent.
	Apply(controls [][]float64)
	// Step advances the environment by dt seconds after controls are applied.
	Step(dt float64)
	// Done reports, per agent, whether its episode has finished.
	Done() []bool
	// Snapshot serializes the world state (JSON recommended).
	Snapshot() ([]byte, error)
	// Restore reinstates a state produced by Snapshot.
	Restore(data []byte) error
}
