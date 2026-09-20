package neat

import (
	"encoding/json"
	"testing"
)

func xorEvaluator(genome *Genome) (float64, error) {
	network, err := NewNetwork(genome)
	if err != nil {
		return 0, err
	}
	patterns := [][3]float64{
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 0},
	}
	outputs := make([]float64, 1)
	errorSum := 0.0
	for _, pattern := range patterns {
		if err := network.Activate([]float64{pattern[0], pattern[1]}, outputs); err != nil {
			return 0, err
		}
		difference := pattern[2] - outputs[0]
		errorSum += difference * difference
	}
	return 1 / (1 + errorSum), nil
}

func xorOptions() Options {
	options := DefaultOptions(2, 1)
	options.PopulationSize = 150
	options.Seed = 1
	options.AddLinkRate = 0.1
	options.AddNodeRate = 0.05
	options.SurvivalRate = 0.2
	options.Elitism = 1
	return options
}

func TestPopulationEpochKeepsSizeAndSpeciates(t *testing.T) {
	population, err := NewPopulation(xorOptions())
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}
	if len(population.Genomes()) != 150 {
		t.Fatalf("initial population = %d", len(population.Genomes()))
	}

	stats, err := population.Epoch(EvaluatorFunc(xorEvaluator), nil)
	if err != nil {
		t.Fatalf("Epoch: %v", err)
	}
	if stats.PopulationSize != 150 {
		t.Fatalf("after epoch population = %d", stats.PopulationSize)
	}
	if stats.SpeciesCount < 1 {
		t.Fatalf("species count = %d", stats.SpeciesCount)
	}
	if population.Generation() != 1 {
		t.Fatalf("generation = %d", population.Generation())
	}
}

func buildXORSolution(t *testing.T) *Genome {
	t.Helper()
	options := xorOptions()
	options.PopulationSize = 60
	population, err := NewPopulation(options)
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}

	const target = 0.95
	for generation := 0; generation < 400; generation++ {
		stats, err := population.Epoch(EvaluatorFunc(xorEvaluator), nil)
		if err != nil {
			t.Fatalf("Epoch %d: %v", generation, err)
		}
		if stats.BestFitness >= target {
			solution := population.BestGenome()
			// Verify the champion really solves XOR.
			network, err := NewNetwork(solution)
			if err != nil {
				t.Fatalf("solution network: %v", err)
			}
			outputs := make([]float64, 1)
			patterns := [][3]float64{{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}
			for _, pattern := range patterns {
				_ = network.Activate([]float64{pattern[0], pattern[1]}, outputs)
				if rounded := outputs[0] >= 0.5; rounded != (pattern[2] == 1) {
					return nil // not actually solved; keep evolving
				}
			}
			return solution
		}
	}
	return nil
}

func TestPopulationSolvesXOR(t *testing.T) {
	solution := buildXORSolution(t)
	if solution == nil {
		t.Fatalf("population did not solve XOR within the generation budget")
	}
	if len(solution.HiddenIDs()) == 0 {
		t.Logf("note: XOR solved without hidden nodes (linear separability is not expected)")
	}
}

func TestPopulationSnapshotResumeIsDeterministic(t *testing.T) {
	options := xorOptions()
	options.PopulationSize = 40
	options.Seed = 99

	original, err := NewPopulation(options)
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}
	for i := 0; i < 5; i++ {
		if _, err := original.Epoch(EvaluatorFunc(xorEvaluator), nil); err != nil {
			t.Fatalf("original epoch: %v", err)
		}
	}

	snapshot, err := original.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	restored, err := DecodePopulation(snapshot)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}

	for i := 0; i < 5; i++ {
		if _, err := original.Epoch(EvaluatorFunc(xorEvaluator), nil); err != nil {
			t.Fatalf("original resume epoch: %v", err)
		}
		if _, err := restored.Epoch(EvaluatorFunc(xorEvaluator), nil); err != nil {
			t.Fatalf("restored epoch: %v", err)
		}
	}

	originalJSON, _ := json.Marshal(original.Genomes())
	restoredJSON, _ := json.Marshal(restored.Genomes())
	if string(originalJSON) != string(restoredJSON) {
		t.Fatalf("resumed population diverged from uninterrupted run")
	}
	if original.Generation() != restored.Generation() {
		t.Fatalf("generations differ: %d vs %d", original.Generation(), restored.Generation())
	}
}
