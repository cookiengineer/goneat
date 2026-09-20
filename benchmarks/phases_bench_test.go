package benchmarks

import "testing"

// phaseComparisonMaxGenerations bounds each task's search budget so the
// comparison finishes in reasonable time.
func phaseComparisonMaxGenerations(task Task) int {
	switch task.Name() {
	case "parity2":
		return 100
	case "parity3":
		return 200
	case "parity4":
		return 300
	case "mux6":
		return 200
	case "cartpole":
		return 50
	case "cubic":
		return 400
	default:
		return 200
	}
}

// BenchmarkPhaseComparison reports, for every standard task and search
// strategy, how many generations were needed, whether the task was solved and
// the best fitness reached. Comparing the sub-benchmarks shows which phase is
// most efficient for a given task.
func BenchmarkPhaseComparison(b *testing.B) {
	tasks := []Task{
		NewParity(2),
		NewParity(3),
		NewParity(4),
		NewMultiplexer(2),
		NewCartPole(500),
		NewCubicBoundary(9),
	}
	seeds := []int64{1, 2, 3}

	for _, task := range tasks {
		task := task
		maxGenerations := phaseComparisonMaxGenerations(task)
		for _, strategy := range Strategies {
			strategy := strategy
			b.Run(task.Name()+"/"+strategy.Name, func(b *testing.B) {
				totalGenerations := 0
				totalFitness := 0.0
				solved := 0

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					config := DefaultRunConfig(task, strategy, seeds[i%len(seeds)], maxGenerations)
					result, err := RunSearch(config)
					if err != nil {
						b.Fatalf("RunSearch: %v", err)
					}
					totalGenerations += result.Generations
					totalFitness += result.BestFitness
					if result.Solved {
						solved++
					}
				}
				b.StopTimer()

				count := float64(b.N)
				b.ReportMetric(float64(totalGenerations)/count, "generations/op")
				b.ReportMetric(float64(solved)/count, "solve_rate")
				b.ReportMetric(totalFitness/count, "best_fitness")
			})
		}
	}
}
