package benchmarks

import (
	"time"

	"github.com/cookiengineer/goneat/engine/neat"
)

// Strategy is one search configuration to compare.
type Strategy struct {
	Name string
	// Enabled turns on phased searching with a cycling schedule.
	Enabled bool
	// Mode is the fixed phase used when Enabled is false.
	Mode neat.Phase
	// Schedule overrides the default {default, prune} schedule when Enabled.
	Schedule []neat.Phase
}

// Strategies are the four phase modes plus the alternating phased search.
var Strategies = []Strategy{
	{Name: "default", Mode: neat.PhaseDefault},
	{Name: "prune", Mode: neat.PhasePrune},
	{Name: "backprop", Mode: neat.PhaseBackprop},
	{Name: "hybrid", Mode: neat.PhaseHybrid},
	{Name: "phased", Enabled: true, Mode: neat.PhaseDefault},
}

// RunConfig parameterizes a single search run.
type RunConfig struct {
	Task            Task
	Strategy        Strategy
	Seed            int64
	PopulationSize  int
	MaxGenerations  int
	AddLinkRate     float64
	AddNodeRate     float64
	SurvivalRate    float64
	PhaseDuration   int
	StallLimit      int
	TrainIterations int
}

// RunResult captures the outcome of a search run.
type RunResult struct {
	Generations int
	Solved      bool
	BestFitness float64
	NodeCount   int
	Elapsed     time.Duration
}

// RunSearch evaluates a population with the given strategy until the task is
// solved or the generation budget is exhausted.
func RunSearch(config RunConfig) (RunResult, error) {
	task := config.Task
	options := neat.DefaultOptions(task.InputCount(), task.OutputCount())
	options.Seed = config.Seed
	options.PopulationSize = config.PopulationSize
	options.AddLinkRate = config.AddLinkRate
	options.AddNodeRate = config.AddNodeRate
	options.SurvivalRate = config.SurvivalRate
	options.Elitism = 1
	options.StagnationLimit = 0
	options.Phases = neat.PhaseOptions{
		Enabled:         config.Strategy.Enabled,
		Mode:            config.Strategy.Mode,
		Schedule:        config.Strategy.Schedule,
		Duration:        config.PhaseDuration,
		StallLimit:      config.StallLimit,
		TrainIterations: config.TrainIterations,
		Trainer:         task.Trainer(),
	}

	population, err := neat.NewPopulation(options)
	if err != nil {
		return RunResult{}, err
	}

	evaluator := neat.EvaluatorFunc(func(genome *neat.Genome) (float64, error) {
		return task.Evaluate(genome)
	})

	start := time.Now()
	result := RunResult{Generations: config.MaxGenerations}
	for generation := 0; generation < config.MaxGenerations; generation++ {
		stats, err := population.Epoch(evaluator, nil)
		if err != nil {
			return RunResult{}, err
		}
		if stats.BestFitness > result.BestFitness {
			result.BestFitness = stats.BestFitness
		}
		if task.Solved(population.BestGenome()) {
			result.Solved = true
			result.Generations = generation + 1
			break
		}
	}
	result.NodeCount = len(population.BestGenome().Nodes)
	result.Elapsed = time.Since(start)
	return result, nil
}

// DefaultRunConfig returns the standard comparison settings for a strategy.
func DefaultRunConfig(task Task, strategy Strategy, seed int64, maxGenerations int) RunConfig {
	return RunConfig{
		Task:            task,
		Strategy:        strategy,
		Seed:            seed,
		PopulationSize:  200,
		MaxGenerations:  maxGenerations,
		AddLinkRate:     0.2,
		AddNodeRate:     0.1,
		SurvivalRate:    0.2,
		PhaseDuration:   15,
		StallLimit:      8,
		TrainIterations: 2,
	}
}
