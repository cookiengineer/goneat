package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/cookiengineer/goneat/agent"
	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/simulation"
)

func TestPongLearnsToTrack(t *testing.T) {
	sim, _, err := NewPongSimulation(16, 400, 7)
	if err != nil {
		t.Fatalf("NewPongSimulation: %v", err)
	}

	// Baseline pool captured before any training.
	baselinePool := make([]*neat.Genome, 0, 8)
	for index, genome := range sim.Population().Genomes() {
		if index >= 8 {
			break
		}
		baselinePool = append(baselinePool, genome.Clone())
	}

	for generation := 0; generation < 25; generation++ {
		if _, err := sim.RunGeneration(context.Background()); err != nil {
			t.Fatalf("RunGeneration %d: %v", generation, err)
		}
	}
	champion := sim.Population().BestGenome()

	evaluationSeeds := []int64{1, 2, 3, 4, 5}
	baselineBest := 0
	for _, genome := range baselinePool {
		total := 0
		for _, seed := range evaluationSeeds {
			total += evaluateGenome(t, genome, 400, seed)
		}
		if total > baselineBest {
			baselineBest = total
		}
	}
	championHits := 0
	for _, seed := range evaluationSeeds {
		championHits += evaluateGenome(t, champion, 400, seed)
	}

	if championHits <= baselineBest {
		t.Fatalf("champion hits (%d) did not beat untrained baseline (%d)", championHits, baselineBest)
	}
}

func evaluateGenome(t *testing.T, genome *neat.Genome, ticks int, seed int64) int {
	t.Helper()
	world := NewPongWorld(1, ticks)
	world.Reset(seed)
	brain, err := agent.NewNEATBrain(genome)
	if err != nil {
		t.Fatalf("NewNEATBrain: %v", err)
	}
	player := agent.NewAgent("champion", brain)
	runner := simulation.InlineRunner{}
	if err := runner.Run(context.Background(), world, []*agent.Agent{player}, simulation.RunConfig{
		Ticks:  ticks,
		DT:     1.0 / 60.0,
		Reward: PongScoreReward,
	}); err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	return world.Hits(0)
}

func TestChampionReplayIsDeterministic(t *testing.T) {
	sim, _, err := NewPongSimulation(12, 400, 21)
	if err != nil {
		t.Fatalf("NewPongSimulation: %v", err)
	}
	for generation := 0; generation < 8; generation++ {
		if _, err := sim.RunGeneration(context.Background()); err != nil {
			t.Fatalf("RunGeneration %d: %v", generation, err)
		}
	}
	champion := sim.Population().BestGenome()

	hitsA := evaluateGenome(t, champion, 400, 1234)
	hitsB := evaluateGenome(t, champion, 400, 1234)
	if hitsA != hitsB {
		t.Fatalf("replay not deterministic: %d vs %d", hitsA, hitsB)
	}

	// The champion must survive serialization and replay identically.
	encoded, err := json.Marshal(champion)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	restored := &neat.Genome{}
	if err := json.Unmarshal(encoded, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	hitsC := evaluateGenome(t, restored, 400, 1234)
	if hitsA != hitsC {
		t.Fatalf("serialized champion replay differs: %d vs %d", hitsA, hitsC)
	}
}

func TestPongSnapshotResumeMatchesUninterrupted(t *testing.T) {
	build := func() (*simulation.Simulation, *neat.Population) {
		sim, _, err := NewPongSimulation(8, 200, 31)
		if err != nil {
			t.Fatalf("NewPongSimulation: %v", err)
		}
		return sim, sim.Population()
	}

	continuous, continuousPopulation := build()
	for i := 0; i < 4; i++ {
		if _, err := continuous.RunGeneration(context.Background()); err != nil {
			t.Fatalf("continuous: %v", err)
		}
	}

	interrupted, _ := build()
	for i := 0; i < 2; i++ {
		if _, err := interrupted.RunGeneration(context.Background()); err != nil {
			t.Fatalf("interrupted: %v", err)
		}
	}
	snapshot, err := interrupted.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	restored, err := simulation.RestoreSimulation(snapshot, NewPongWorld(8, 200))
	if err != nil {
		t.Fatalf("RestoreSimulation: %v", err)
	}
	restored.SetReward(PongReward)
	restored.SetRunner(simulation.InlineRunner{})
	for i := 0; i < 2; i++ {
		if _, err := restored.RunGeneration(context.Background()); err != nil {
			t.Fatalf("restored: %v", err)
		}
	}

	continuousJSON, _ := json.Marshal(continuousPopulation.Genomes())
	restoredJSON, _ := json.Marshal(restored.Population().Genomes())
	if string(continuousJSON) != string(restoredJSON) {
		t.Fatalf("restored Pong run diverged from uninterrupted run")
	}
}
