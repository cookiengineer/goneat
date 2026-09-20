package simulation

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/goneat/agent"
	"github.com/cookiengineer/goneat/engine/neat"
)

// testWorld is a deterministic environment: each agent controls one value that
// moves toward a target. It is intentionally simple so runner equivalence and
// snapshot resumption are easy to assert.
type testWorld struct {
	sensorDimensions  int
	controlDimensions int
	agentCount        int
	maxTicks          int

	Target    float64
	positions []float64
	ticks     int
	seedUsed  int64
}

type testWorldState struct {
	Positions []float64 `json:"positions"`
	Ticks     int       `json:"ticks"`
	SeedUsed  int64     `json:"seed_used"`
}

func newTestWorld(agentCount, sensorDimensions, controlDimensions, maxTicks int) *testWorld {
	return &testWorld{
		sensorDimensions:  sensorDimensions,
		controlDimensions: controlDimensions,
		agentCount:        agentCount,
		maxTicks:          maxTicks,
		Target:            0.5,
		positions:         make([]float64, agentCount),
	}
}

func (w *testWorld) Reset(seed int64) {
	w.ticks = 0
	w.seedUsed = seed
	for i := range w.positions {
		w.positions[i] = float64((seed+int64(i*131))%100) / 100.0
	}
}

func (w *testWorld) SensorDimensions() int  { return w.sensorDimensions }
func (w *testWorld) ControlDimensions() int { return w.controlDimensions }

func (w *testWorld) Sensors() [][]float64 {
	frames := make([][]float64, w.agentCount)
	for i := range frames {
		frame := make([]float64, w.sensorDimensions)
		frame[0] = w.positions[i]
		frames[i] = frame
	}
	return frames
}

func (w *testWorld) Apply(controls [][]float64) {
	for i := range w.positions {
		if i >= len(controls) || len(controls[i]) == 0 {
			continue
		}
		w.positions[i] += 0.1 * controls[i][0]
		if w.positions[i] > 1 {
			w.positions[i] = 1
		}
		if w.positions[i] < -1 {
			w.positions[i] = -1
		}
	}
}

func (w *testWorld) Step(float64) { w.ticks++ }

func (w *testWorld) Done() []bool {
	done := make([]bool, w.agentCount)
	for i := range done {
		done[i] = w.ticks >= w.maxTicks
	}
	return done
}

func (w *testWorld) Snapshot() ([]byte, error) {
	return json.Marshal(testWorldState{Positions: w.positions, Ticks: w.ticks, SeedUsed: w.seedUsed})
}

func (w *testWorld) Restore(data []byte) error {
	var state testWorldState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	w.positions = state.Positions
	w.ticks = state.Ticks
	w.seedUsed = state.SeedUsed
	return nil
}

func smallPopulation(seed int64) *neat.Population {
	options := neat.DefaultOptions(1, 1)
	options.PopulationSize = 8
	options.Seed = seed
	options.AddLinkRate = 0.0
	options.AddNodeRate = 0.0
	options.ToggleEnableRate = 0.0
	options.StagnationLimit = 0
	population, err := neat.NewPopulation(options)
	if err != nil {
		panic(err)
	}
	return population
}

func closenessReward(step *StepContext, a *agent.Agent) (float64, *agent.Training) {
	index := step.AgentIndex(a.ID)
	if index < 0 || index >= len(step.Sensors) {
		return 0, nil
	}
	world := step.World.(*testWorld)
	distance := math.Abs(step.Sensors[index][0] - world.Target)
	return -distance, nil
}

func TestRunnersProduceIdenticalResults(t *testing.T) {
	run := func(runner Runner) (string, string) {
		population := smallPopulation(2024)
		world := newTestWorld(population.Options().PopulationSize, 1, 1, 20)
		simulation, err := NewSimulation(world, population, Options{
			Ticks:  20,
			DT:     0.1,
			Seed:   7,
			Runner: runner,
			Reward: closenessReward,
		})
		if err != nil {
			t.Fatalf("NewSimulation: %v", err)
		}
		if _, err := simulation.RunGeneration(context.Background()); err != nil {
			t.Fatalf("RunGeneration: %v", err)
		}
		genomes, _ := json.Marshal(population.Genomes())
		worldState, _ := world.Snapshot()
		return string(genomes), string(worldState)
	}

	inlineGenomes, inlineWorld := run(InlineRunner{})
	channelGenomes, channelWorld := run(ChannelRunner{})

	if inlineGenomes != channelGenomes {
		t.Fatalf("InlineRunner and ChannelRunner produced different genomes")
	}
	if inlineWorld != channelWorld {
		t.Fatalf("InlineRunner and ChannelRunner produced different world states")
	}
}

func TestCallbacksAdjustFitnessAndTrain(t *testing.T) {
	options := neat.DefaultOptions(1, 1)
	options.PopulationSize = 4
	options.Seed = 5
	population, err := neat.NewPopulation(options)
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}
	initialWeights := population.Genomes()[0].WeightVector()

	world := newTestWorld(4, 1, 1, 10)
	simulation, err := NewSimulation(world, population, Options{
		Ticks:  10,
		DT:     0.1,
		Seed:   3,
		Runner: InlineRunner{},
		Reward: func(step *StepContext, a *agent.Agent) (float64, *agent.Training) {
			index := step.AgentIndex(a.ID)
			if index < 0 {
				return 0, nil
			}
			training := &agent.Training{
				Inputs:     append([]float64(nil), step.Sensors[index]...),
				Targets:    []float64{1},
				Rate:       0.5,
				Iterations: 1,
			}
			return 0.5, training
		},
	})
	if err != nil {
		t.Fatalf("NewSimulation: %v", err)
	}

	if err := simulation.RunEpisode(context.Background()); err != nil {
		t.Fatalf("RunEpisode: %v", err)
	}

	for _, a := range []int{0, 1, 2, 3} {
		if population.Genomes()[a].Fitness == 0 {
			t.Fatalf("genome %d fitness was not updated", a)
		}
	}
	finalWeights := population.Genomes()[0].WeightVector()
	changed := false
	for i := range initialWeights {
		if math.Abs(initialWeights[i]-finalWeights[i]) > 1e-12 {
			changed = true
		}
	}
	if !changed {
		t.Fatalf("training sample did not change weights")
	}
}

func TestSnapshotResumeMatchesUninterrupted(t *testing.T) {
	build := func() (*Simulation, *testWorld, *neat.Population) {
		population := smallPopulation(4242)
		world := newTestWorld(population.Options().PopulationSize, 1, 1, 15)
		simulation, err := NewSimulation(world, population, Options{
			Ticks:  15,
			DT:     0.1,
			Seed:   11,
			Runner: InlineRunner{},
			Reward: closenessReward,
		})
		if err != nil {
			t.Fatalf("NewSimulation: %v", err)
		}
		return simulation, world, population
	}

	// Uninterrupted: 4 generations.
	continuous, _, continuousPopulation := build()
	for i := 0; i < 4; i++ {
		if _, err := continuous.RunGeneration(context.Background()); err != nil {
			t.Fatalf("continuous: %v", err)
		}
	}

	// Interrupted: 2 generations, snapshot, restore, 2 more.
	interrupted, _, _ := build()
	for i := 0; i < 2; i++ {
		if _, err := interrupted.RunGeneration(context.Background()); err != nil {
			t.Fatalf("interrupted: %v", err)
		}
	}
	snapshot, err := interrupted.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	freshWorld := newTestWorld(smallPopulation(4242).Options().PopulationSize, 1, 1, 15)
	restored, err := RestoreSimulation(snapshot, freshWorld)
	if err != nil {
		t.Fatalf("RestoreSimulation: %v", err)
	}
	restored.SetReward(closenessReward)
	restored.SetRunner(InlineRunner{})
	for i := 0; i < 2; i++ {
		if _, err := restored.RunGeneration(context.Background()); err != nil {
			t.Fatalf("restored: %v", err)
		}
	}

	continuousJSON, _ := json.Marshal(continuousPopulation.Genomes())
	restoredJSON, _ := json.Marshal(restored.Population().Genomes())
	if string(continuousJSON) != string(restoredJSON) {
		t.Fatalf("restored run diverged from uninterrupted run")
	}
	if continuous.Generation() != restored.Generation() {
		t.Fatalf("generations differ: %d vs %d", continuous.Generation(), restored.Generation())
	}
}

func TestAutoSaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "pong_snapshot.json")

	population := smallPopulation(777)
	world := newTestWorld(population.Options().PopulationSize, 1, 1, 10)
	simulation, err := NewSimulation(world, population, Options{
		Ticks:         10,
		DT:            0.1,
		Seed:          1,
		Runner:        InlineRunner{},
		Reward:        closenessReward,
		AutoSavePath:  path,
		AutoSaveEvery: 1,
	})
	if err != nil {
		t.Fatalf("NewSimulation: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := simulation.RunGeneration(context.Background()); err != nil {
			t.Fatalf("RunGeneration %d: %v", i, err)
		}
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("autosave file missing: %v", err)
	}

	restored, err := LoadSimulation(path, newTestWorld(population.Options().PopulationSize, 1, 1, 10))
	if err != nil {
		t.Fatalf("LoadSimulation: %v", err)
	}
	if restored.Generation() != 3 {
		t.Fatalf("restored generation = %d, want 3", restored.Generation())
	}
	if restored.AutoSavePath() != path {
		t.Fatalf("autosave path not preserved: %q", restored.AutoSavePath())
	}
	restored.SetReward(closenessReward)
	restored.SetRunner(InlineRunner{})
	if _, err := restored.RunGeneration(context.Background()); err != nil {
		t.Fatalf("restored RunGeneration: %v", err)
	}
	if restored.Generation() != 4 {
		t.Fatalf("restored generation after resume = %d, want 4", restored.Generation())
	}
}
