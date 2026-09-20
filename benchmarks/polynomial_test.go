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
	if quintic.Boundary(-0.5) != -math.Pow(0.5, 5) {
		t.Fatalf("quintic boundary(-0.5) = %v", quintic.Boundary(-0.5))
	}

	septic := NewSepticBoundary(11)
	if septic.Boundary(0.5) != math.Pow(0.5, 7) {
		t.Fatalf("septic boundary(0.5) = %v", septic.Boundary(0.5))
	}
	if septic.Boundary(-0.5) != -math.Pow(0.5, 7) {
		t.Fatalf("septic boundary(-0.5) = %v", septic.Boundary(-0.5))
	}
}

// TestOddPowerBoundariesAreDistinct guards the higher-order tasks: the tuned
// grid must actually separate y = x^3, x^5 and x^7, otherwise x^5 and x^7
// collapse to the same labeled dataset and the benchmarks measure nothing new.
func TestOddPowerBoundariesAreDistinct(t *testing.T) {
	tasks := map[string]*PolynomialClassification{
		"cubic":   NewOddPowerBoundary("cubic", 3),
		"quintic": NewOddPowerBoundary("quintic", 5),
		"septic":  NewOddPowerBoundary("septic", 7),
	}
	labels := func(task *PolynomialClassification) []float64 {
		points := task.Points()
		values := make([]float64, len(points))
		for i, point := range points {
			values[i] = point.Targets[0]
		}
		return values
	}
	orders := []string{"cubic", "quintic", "septic"}
	for i := 0; i < len(orders); i++ {
		for j := i + 1; j < len(orders); j++ {
			left := labels(tasks[orders[i]])
			right := labels(tasks[orders[j]])
			if len(left) != len(right) {
				t.Fatalf("%s and %s have different point counts (%d vs %d)",
					orders[i], orders[j], len(left), len(right))
			}
			identical := true
			for k := range left {
				if left[k] != right[k] {
					identical = false
					break
				}
			}
			if identical {
				t.Fatalf("%s and %s produce identical labels on the tuned grid", orders[i], orders[j])
			}
		}
	}
}

// TestPolynomialClassificationSolvable proves the curved, odd-degree decision
// boundaries are learnable. It also proves per-node activation functions are
// used: a solved champion must contain hidden nodes with more than one distinct
// activation.
func TestPolynomialClassificationSolvable(t *testing.T) {
	cases := []struct {
		task   *PolynomialClassification
		budget int
	}{
		{NewCubicBoundary(11), 500},
		{NewOddPowerBoundary("quintic", 5), 300},
		{NewOddPowerBoundary("septic", 7), 300},
	}
	for _, testCase := range cases {
		task := testCase.task
		t.Run(task.Name(), func(t *testing.T) {
			config := DefaultRunConfig(task, Strategy{Name: "default", Mode: neat.PhaseDefault}, 1, testCase.budget)
			result, err := RunSearch(config)
			if err != nil {
				t.Fatalf("RunSearch: %v", err)
			}
			if !result.Solved {
				t.Fatalf("%s classification not solved within %d generations (best %.4f)",
					task.Name(), config.MaxGenerations, result.BestFitness)
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
		})
	}
}
