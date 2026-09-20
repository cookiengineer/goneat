package backprop

import (
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/engine/random"
)

// These benchmarks make training efficiency measurable over time. The
// convergence benchmark reports ns/op (time to reach a solution) plus custom
// metrics: iterations to solve, final loss and solve rate. Comparing the
// sub-benchmarks shows how different error functions and optimizers affect the
// outcome rather than only the raw speed.

// BenchmarkXORLearnStep measures the cost of a single gradient step, which is
// the building block every training loop is made of.
func BenchmarkXORLearnStep(b *testing.B) {
	generator := random.New(7)
	weights := make([]float64, xorHiddenWeightCount(4))
	for i := range weights {
		weights[i] = generator.NormFloat64()
	}
	genome := xorHiddenGenome(4, weights)
	network, err := neat.NewNetwork(genome)
	if err != nil {
		b.Fatalf("NewNetwork: %v", err)
	}
	trainer := NewTrainer(NewSGD(0.5, 0.9), 10)
	loss := MSELoss{}
	inputs := []float64{0, 1}
	targets := []float64{1}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := trainer.Learn(genome, network, inputs, targets, loss); err != nil {
			b.Fatalf("Learn: %v", err)
		}
	}
}

// BenchmarkXORConvergence measures the full time-to-solution and reports
// training quality metrics for each loss/optimizer pairing.
func BenchmarkXORConvergence(b *testing.B) {
	for _, base := range convergingConfigs() {
		base := base
		b.Run(base.Name, func(b *testing.B) {
			totalIterations := 0
			totalLoss := 0.0
			solved := 0

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				config := base
				config.Hidden = 4
				config.Seed = int64(i%10 + 1)
				config.MaxPasses = 2000
				config.StopOnSolve = true

				result := trainXOR(config)
				totalIterations += result.Iterations
				totalLoss += result.FinalLoss
				if result.Solved {
					solved++
				}
			}
			b.StopTimer()

			count := float64(b.N)
			b.ReportMetric(float64(totalIterations)/count, "iters/op")
			b.ReportMetric(totalLoss/count, "final_loss")
			b.ReportMetric(float64(solved)/count, "solve_rate")
		})
	}
}

// BenchmarkXORFixedBudget runs an equal number of passes without early
// stopping. It is the fairest way to compare two error functions: same cost,
// compare resulting loss and accuracy.
func BenchmarkXORFixedBudget(b *testing.B) {
	for _, base := range convergingConfigs() {
		base := base
		b.Run(base.Name, func(b *testing.B) {
			config := base
			config.Hidden = 4
			config.Seed = 7
			config.MaxPasses = 200
			config.StopOnSolve = false

			var last xorResult
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				last = trainXOR(config)
			}
			b.StopTimer()

			b.ReportMetric(last.FinalLoss, "final_loss")
			b.ReportMetric(last.Accuracy, "accuracy")
		})
	}
}
