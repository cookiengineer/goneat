package main

import (
	"strings"
	"testing"

	"github.com/cookiengineer/goneat/benchmarks"
	"github.com/cookiengineer/goneat/engine/neat"
)

// TestParabolaDemoImproves checks that evolution improves the classification
// fitness and grows hidden nodes. Full convergence is covered by the benchmark
// suite; this keeps the example test quick.
func TestParabolaDemoImproves(t *testing.T) {
	task := benchmarks.NewCubicBoundary(11)
	population, err := buildPopulation(1)
	if err != nil {
		t.Fatalf("buildPopulation: %v", err)
	}

	first, err := population.Epoch(neat.EvaluatorFunc(task.Evaluate), nil)
	if err != nil {
		t.Fatalf("first epoch: %v", err)
	}
	var last = first
	for generation := 1; generation < 120; generation++ {
		stats, err := population.Epoch(neat.EvaluatorFunc(task.Evaluate), nil)
		if err != nil {
			t.Fatalf("epoch %d: %v", generation, err)
		}
		last = stats
	}

	if last.BestFitness <= first.BestFitness {
		t.Fatalf("fitness did not improve: first %.4f, last %.4f", first.BestFitness, last.BestFitness)
	}
	if hidden := len(population.BestGenome().HiddenIDs()); hidden == 0 {
		t.Fatalf("expected hidden nodes to be added")
	}
}

func TestRenderGrid(t *testing.T) {
	task := benchmarks.NewCubicBoundary(11)
	population, err := buildPopulation(2)
	if err != nil {
		t.Fatalf("buildPopulation: %v", err)
	}
	grid := renderGrid(task, population.Genomes()[0], 21, 11)
	if grid == "" {
		t.Fatalf("renderGrid returned empty output")
	}
	if lines := strings.Count(grid, "\n"); lines != 11 {
		t.Fatalf("renderGrid produced %d lines, want 11", lines)
	}
}
