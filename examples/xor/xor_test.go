package main

import (
	"encoding/json"
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
)

func TestXORExampleSolves(t *testing.T) {
	options := neat.DefaultOptions(2, 1)
	options.Seed = 1
	population, err := neat.NewPopulation(options)
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}

	solved := false
	for generation := 0; generation < 300; generation++ {
		if _, err := population.Epoch(neat.EvaluatorFunc(XORFitness), nil); err != nil {
			t.Fatalf("Epoch %d: %v", generation, err)
		}
		if XORSolved(population.BestGenome()) {
			solved = true
			break
		}
	}
	if !solved {
		t.Fatalf("XOR was not solved within the generation budget")
	}

	// The champion must survive a JSON round trip.
	champion := population.BestGenome()
	encoded, err := json.Marshal(champion)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	restored := &neat.Genome{}
	if err := json.Unmarshal(encoded, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !XORSolved(restored) {
		t.Fatalf("restored champion no longer solves XOR")
	}
}
