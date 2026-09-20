// Command xor evolves a NEAT population to solve the XOR function. It
// demonstrates the core genetic engine without any simulation wiring.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/cookiengineer/goneat/engine/neat"
)

// XORPatterns are the four ground-truth cases.
var XORPatterns = [][3]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

// XORFitness scores a genome by how well it reproduces XOR. A perfect network
// scores 1.0.
func XORFitness(genome *neat.Genome) (float64, error) {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return 0, err
	}
	outputs := make([]float64, network.OutputCount())
	errorSum := 0.0
	for _, pattern := range XORPatterns {
		if err := network.Activate([]float64{pattern[0], pattern[1]}, outputs); err != nil {
			return 0, err
		}
		difference := pattern[2] - outputs[0]
		errorSum += difference * difference
	}
	return 1 / (1 + errorSum), nil
}

// XORSolved reports whether a genome classifies all four patterns correctly.
func XORSolved(genome *neat.Genome) bool {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return false
	}
	outputs := make([]float64, network.OutputCount())
	for _, pattern := range XORPatterns {
		if err := network.Activate([]float64{pattern[0], pattern[1]}, outputs); err != nil {
			return false
		}
		predicted := outputs[0] >= 0.5
		if predicted != (pattern[2] == 1) {
			return false
		}
	}
	return true
}

func main() {
	options := neat.DefaultOptions(2, 1)
	options.Seed = 1
	population, err := neat.NewPopulation(options)
	if err != nil {
		log.Fatal(err)
	}

	observer := neat.ObserverFunc(func(stats neat.GenerationStats) {
		fmt.Printf("gen %3d | species %2d | best %.4f | avg %.4f\n",
			stats.Generation, stats.SpeciesCount, stats.BestFitness, stats.AverageFitness)
	})

	for generation := 0; generation < 300; generation++ {
		if _, err := population.Epoch(neat.EvaluatorFunc(XORFitness), observer); err != nil {
			log.Fatal(err)
		}
		// Periodically persist the population to the current working directory.
		if population.Generation()%10 == 0 {
			if data, err := population.Encode(); err == nil {
				if err := os.WriteFile("xor_population.json", data, 0o644); err != nil {
					log.Fatal(err)
				}
			}
		}
		if XORSolved(population.BestGenome()) {
			fmt.Printf("solved in generation %d\n", population.Generation())
			break
		}
	}

	champion := population.BestGenome()
	encoded, err := json.MarshalIndent(champion, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(encoded))
}
