package benchmarks

import (
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
)

// complexifyingStrategies can add hidden structure, which tasks like parity
// require. Pure prune and pure backprop phases cannot, which is itself a
// comparative result the benchmarks surface.
var complexifyingStrategies = []string{"default", "hybrid", "phased"}

func strategyByName(t *testing.T, name string) Strategy {
	t.Helper()
	for _, strategy := range Strategies {
		if strategy.Name == name {
			return strategy
		}
	}
	t.Fatalf("unknown strategy %q", name)
	return Strategy{}
}

func TestParity2SolvableByComplexifyingStrategies(t *testing.T) {
	task := NewParity(2)
	for _, name := range complexifyingStrategies {
		strategy := strategyByName(t, name)
		t.Run(name, func(t *testing.T) {
			config := DefaultRunConfig(task, strategy, 1, 150)
			result, err := RunSearch(config)
			if err != nil {
				t.Fatalf("RunSearch: %v", err)
			}
			if !result.Solved {
				t.Fatalf("%s did not solve parity2 within %d generations (best %.4f)",
					name, config.MaxGenerations, result.BestFitness)
			}
		})
	}
}

func TestCartPoleSolvableByDefault(t *testing.T) {
	task := NewCartPole(500)
	config := DefaultRunConfig(task, Strategy{Name: "default", Mode: neat.PhaseDefault}, 1, 30)
	result, err := RunSearch(config)
	if err != nil {
		t.Fatalf("RunSearch: %v", err)
	}
	if !result.Solved {
		t.Fatalf("default did not solve cartpole within %d generations (best %.1f)",
			config.MaxGenerations, result.BestFitness)
	}
}

func TestAllStrategiesRunAndProduceResults(t *testing.T) {
	task := NewParity(4)
	for _, strategy := range Strategies {
		config := DefaultRunConfig(task, strategy, 2, 5)
		result, err := RunSearch(config)
		if err != nil {
			t.Fatalf("%s: %v", strategy.Name, err)
		}
		if result.BestFitness <= 0 || result.NodeCount == 0 {
			t.Fatalf("%s produced unusable result: %+v", strategy.Name, result)
		}
	}
}
