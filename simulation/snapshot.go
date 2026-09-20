package simulation

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/cookiengineer/goneat/agent"
	"github.com/cookiengineer/goneat/engine/neat"
)

const (
	snapshotFormat  = "goneat.simulation"
	snapshotVersion = 1
)

type snapshotState struct {
	Format        string          `json:"format"`
	Version       int             `json:"version"`
	Seed          int64           `json:"seed"`
	Generation    int             `json:"generation"`
	Ticks         int             `json:"ticks"`
	DT            float64         `json:"dt"`
	AutoSavePath  string          `json:"auto_save_path,omitempty"`
	AutoSaveEvery int             `json:"auto_save_every,omitempty"`
	World         json.RawMessage `json:"world"`
	Population    json.RawMessage `json:"population"`
}

// Snapshot serializes the simulation: the world's own state plus the full
// genome population (including RNG and innovation state). Partial training
// phases can be saved and resumed exactly with RestoreSimulation.
func (s *Simulation) Snapshot() ([]byte, error) {
	worldState, err := s.world.Snapshot()
	if err != nil {
		return nil, fmt.Errorf("simulation: world snapshot: %w", err)
	}
	populationState, err := s.population.Encode()
	if err != nil {
		return nil, fmt.Errorf("simulation: population snapshot: %w", err)
	}
	return json.Marshal(snapshotState{
		Format:        snapshotFormat,
		Version:       snapshotVersion,
		Seed:          s.seed,
		Generation:    s.generation,
		Ticks:         s.ticks,
		DT:            s.dt,
		AutoSavePath:  s.autoSavePath,
		AutoSaveEvery: s.autoSaveEvery,
		World:         worldState,
		Population:    populationState,
	})
}

// SaveTo writes a snapshot to a file, creating or truncating it.
func (s *Simulation) SaveTo(path string) error {
	data, err := s.Snapshot()
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadSimulation reads a snapshot file and restores it into the supplied
// world.
func LoadSimulation(path string, world World) (*Simulation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return RestoreSimulation(data, world)
}

// RestoreSimulation rebuilds a simulation from a snapshot. The caller supplies
// a fresh World of the correct concrete type; its Restore method reinstates the
// saved state. Reward/Punish callbacks are not serialized and must be
// re-registered by the caller.
func RestoreSimulation(data []byte, world World) (*Simulation, error) {
	if world == nil {
		return nil, errors.New("simulation: nil world for restore")
	}
	var state snapshotState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, err
	}
	if state.Format != snapshotFormat {
		return nil, fmt.Errorf("simulation: unexpected snapshot format %q", state.Format)
	}
	if state.Version != snapshotVersion {
		return nil, fmt.Errorf("simulation: unsupported snapshot version %d", state.Version)
	}

	if err := world.Restore(state.World); err != nil {
		return nil, fmt.Errorf("simulation: world restore: %w", err)
	}
	population, err := neat.DecodePopulation(state.Population)
	if err != nil {
		return nil, fmt.Errorf("simulation: population restore: %w", err)
	}

	simulation := &Simulation{
		world:         world,
		population:    population,
		runner:        ChannelRunner{},
		ticks:         state.Ticks,
		dt:            state.DT,
		seed:          state.Seed,
		generation:    state.Generation,
		autoSavePath:  state.AutoSavePath,
		autoSaveEvery: state.AutoSaveEvery,
	}
	if simulation.ticks <= 0 {
		simulation.ticks = 1000
	}
	if simulation.dt <= 0 {
		simulation.dt = 1.0 / 60.0
	}
	simulation.brainFactory = func(genome *neat.Genome) (agent.Brain, error) {
		return agent.NewNEATBrain(genome)
	}
	return simulation, nil
}
