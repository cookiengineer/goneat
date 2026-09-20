package benchmarks

import (
	"math"
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
)

func TestPolynomialBoundaryLabels(t *testing.T) {
	task := NewCubicBoundary(11)
	if task.Boundary(0.5) != 0.125 {
		t.Fatalf("cubic boundary(0.5) = %v, want 0.125", task.Boundary(0.5))
	}
	if task.Boundary(-0.5) != -0.125 {
		t.Fatalf("cubic boundary(-0.5) = %v, want -0.125", task.Boundary(-0.5))
	}
	if len(task.Points()) != 11*11 {
		t.Fatalf("points = %d, want 121", len(task.Points()))
	}
	for _, sample := range task.Points() {
		expected := 0.0
		if sample.Inputs[1] >= task.Boundary(sample.Inputs[0]) {
			expected = 1
		}
		if sample.Targets[0] != expected {
			t.Fatalf("point %v labeled %v, want %v", sample.Inputs, sample.Targets[0], expected)
		}
	}

	quintic := NewQuinticBoundary(11)
	if quintic.Boundary(0.5) != math.Pow(0.5, 5) {
		t.Fatalf("quintic boundary(0.5) = %v", quintic.Boundary(0.5))
	}
}

// TestPolynomialClassificationSolvable proves the curved, odd-degree decision
// boundary is learnable. It also proves per-node activation functions are used:
// a solved champion must contain hidden nodes with more than one distinct
// activation.
func TestPolynomialClassificationSolvable(t *testing.T) {
	task := NewCubicBoundary(11)
	config := DefaultRunConfig(task, Strategy{Name: "default", Mode: neat.PhaseDefault}, 1, 500)
	result, err := RunSearch(config)
	if err != nil {
		t.Fatalf("RunSearch: %v", err)
	}
	if !result.Solved {
		t.Fatalf("cubic classification not solved within %d generations (best %.4f)",
			config.MaxGenerations, result.BestFitness)
	}
	if result.Champion == nil {
		t.Fatalf("no champion returned")
	}
	if hidden := len(result.Champion.HiddenIDs()); hidden < 2 {
		t.Fatalf("expected at least 2 hidden nodes after expansion, got %d", hidden)
	}
	if distinct := result.Champion.DistinctHiddenActivations(); distinct < 2 {
		t.Fatalf("expected hidden nodes with different activation functions, got %d", distinct)
	}
}
