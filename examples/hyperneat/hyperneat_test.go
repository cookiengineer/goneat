package main

import (
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
)

func TestRetinaFitnessInRange(t *testing.T) {
	options := neat.DefaultOptions(6, 1)
	options.Seed = 5
	options.OutputActivation = neat.ActivationTanh
	population, err := neat.NewPopulation(options)
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}
	fitness, err := RetinaFitness(population.Genomes()[0])
	if err != nil {
		t.Fatalf("RetinaFitness: %v", err)
	}
	if fitness < 0 || fitness > 1 {
		t.Fatalf("fitness out of range: %v", fitness)
	}
}

func TestEvolvableSubstrateHyperNEATImproves(t *testing.T) {
	options := neat.DefaultOptions(6, 1)
	options.Seed = 1
	options.OutputActivation = neat.ActivationTanh
	options.AddLinkRate = 0.12
	options.AddNodeRate = 0.06
	options.PopulationSize = 24
	options.StagnationLimit = 0

	population, err := neat.NewPopulation(options)
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}

	first := 0.0
	best := 0.0
	for generation := 0; generation < 25; generation++ {
		stats, err := population.Epoch(neat.EvaluatorFunc(RetinaFitness), nil)
		if err != nil {
			t.Fatalf("Epoch %d: %v", generation, err)
		}
		if generation == 0 {
			first = stats.BestFitness
		}
		if stats.BestFitness > best {
			best = stats.BestFitness
		}
	}

	if best <= first {
		t.Fatalf("fitness did not improve: first %.4f, best %.4f", first, best)
	}
}
