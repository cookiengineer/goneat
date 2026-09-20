package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/simulation"
)

// PongOptions returns the NEAT options for the Pong population.
func PongOptions(populationSize int, seed int64) neat.Options {
	options := neat.DefaultOptions(pongSensorCount, 1)
	options.OutputActivation = neat.ActivationTanh
	options.PopulationSize = populationSize
	options.Seed = seed
	options.AddLinkRate = 0.1
	options.AddNodeRate = 0.05
	options.SurvivalRate = 0.25
	options.Elitism = 1
	options.StagnationLimit = 0
	return options
}

// NewPongSimulation builds a ready-to-run Pong training simulation.
func NewPongSimulation(populationSize, maxTicks int, seed int64) (*simulation.Simulation, *PongWorld, error) {
	population, err := neat.NewPopulation(PongOptions(populationSize, seed))
	if err != nil {
		return nil, nil, err
	}
	world := NewPongWorld(populationSize, maxTicks)
	sim, err := simulation.NewSimulation(world, population, simulation.Options{
		Ticks:  maxTicks,
		DT:     1.0 / 60.0,
		Seed:   seed,
		Runner: simulation.ChannelRunner{},
		Reward: PongReward,
	})
	if err != nil {
		return nil, nil, err
	}
	return sim, world, nil
}

func main() {
	sim, world, err := NewPongSimulation(16, 600, 7)
	if err != nil {
		log.Fatal(err)
	}
	// Periodically persist the whole simulation to the current working
	// directory so training can be inspected or resumed.
	sim.SetAutoSave("pong_snapshot.json", 5)

	sim.SetObserver(neat.ObserverFunc(func(stats neat.GenerationStats) {
		good, evil := world.Scores()
		fmt.Printf("gen %3d | species %2d | best %7.1f | avg %7.1f | score %d - %d\n",
			stats.Generation, stats.SpeciesCount, stats.BestFitness, stats.AverageFitness, good, evil)
	}))

	if _, err := sim.RunGenerations(context.Background(), 30); err != nil {
		log.Fatal(err)
	}

	champion := sim.Population().BestGenome()
	encoded, err := json.MarshalIndent(champion, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("champion fitness %.1f\n", champion.Fitness)
	fmt.Println(string(encoded))
}
