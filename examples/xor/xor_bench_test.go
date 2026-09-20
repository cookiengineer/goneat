package main

import (
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
)

// BenchmarkPopulationSolvesXOR measures the evolutionary (topology) side of
// training: how much wall-clock time and how many generations the NEAT
// population needs to solve XOR. It complements the backprop benchmarks in
// engine/backprop.
func BenchmarkPopulationSolvesXOR(b *testing.B) {
	options := neat.DefaultOptions(2, 1)
	options.Seed = 1
	options.PopulationSize = 150
	options.AddLinkRate = 0.1
	options.AddNodeRate = 0.05
	options.SurvivalRate = 0.2
	options.Elitism = 1

	const maxGenerations = 400

	totalGenerations := 0
	solved := 0

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		population, err := neat.NewPopulation(options)
		if err != nil {
			b.Fatalf("NewPopulation: %v", err)
		}
		generation := 0
		for ; generation < maxGenerations; generation++ {
			if _, err := population.Epoch(neat.EvaluatorFunc(XORFitness), nil); err != nil {
				b.Fatalf("Epoch: %v", err)
			}
			if XORSolved(population.BestGenome()) {
				solved++
				break
			}
		}
		totalGenerations += generation
	}
	b.StopTimer()

	b.ReportMetric(float64(totalGenerations)/float64(b.N), "generations/op")
	b.ReportMetric(float64(solved)/float64(b.N), "solve_rate")
}
