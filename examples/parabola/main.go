// Command parabola demonstrates 2D point classification with a curved,
// odd-degree decision boundary. Points (x, y) are labeled positive when they
// lie above the curve y = x^3. A straight line cannot separate the two classes,
// so the population must grow hidden nodes and use several per-node activation
// functions to bend the boundary.
//
// The demo prints the evolved activation histogram and an ASCII rendering of
// the learned boundary, and auto-saves the population to the working
// directory.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/cookiengineer/goneat/benchmarks"
	"github.com/cookiengineer/goneat/engine/neat"
)

const (
	gridResolution = 21
	maxGenerations = 600
	autoSaveEvery  = 25
)

// buildPopulation creates the NEAT population for the classification task.
func buildPopulation(seed int64) (*neat.Population, error) {
	options := neat.DefaultOptions(2, 1)
	options.Seed = seed
	options.PopulationSize = 200
	options.AddLinkRate = 0.2
	options.AddNodeRate = 0.1
	options.SurvivalRate = 0.2
	options.Elitism = 1
	options.StagnationLimit = 0
	return neat.NewPopulation(options)
}

// train evolves a population and returns the best genome. When savePath is
// non-empty the population is auto-saved there every autoSaveEvery generations.
func train(task *benchmarks.PolynomialClassification, seed int64, generations int, savePath string) (*neat.Population, *neat.Genome, bool) {
	population, err := buildPopulation(seed)
	if err != nil {
		log.Fatal(err)
	}

	solved := false
	for generation := 0; generation < generations; generation++ {
		stats, err := population.Epoch(neat.EvaluatorFunc(task.Evaluate), nil)
		if err != nil {
			log.Fatal(err)
		}
		if savePath != "" && (generation+1)%autoSaveEvery == 0 {
			if data, err := population.Encode(); err == nil {
				_ = os.WriteFile(savePath, data, 0o644)
			}
		}
		if stats.Generation%50 == 0 {
			fmt.Printf("gen %3d | species %2d | best %.4f | avg %.4f\n",
				stats.Generation, stats.SpeciesCount, stats.BestFitness, stats.AverageFitness)
		}
		if task.Solved(population.BestGenome()) {
			solved = true
			break
		}
	}
	return population, population.BestGenome(), solved
}

// renderGrid draws the learned decision region. '#' is a positive prediction,
// '.' a negative one, and '!' a misclassification.
func renderGrid(task *benchmarks.PolynomialClassification, genome *neat.Genome, width, height int) string {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return ""
	}
	outputs := make([]float64, 1)
	rendered := ""
	for row := 0; row < height; row++ {
		y := 1 - 2*float64(row)/float64(height-1)
		line := make([]byte, 0, width+1)
		for col := 0; col < width; col++ {
			x := -1 + 2*float64(col)/float64(width-1)
			if err := network.Activate([]float64{x, y}, outputs); err != nil {
				continue
			}
			predicted := outputs[0] >= 0.5
			actual := y >= task.Boundary(x)
			switch {
			case predicted == actual && predicted:
				line = append(line, '#')
			case predicted == actual:
				line = append(line, '.')
			default:
				line = append(line, '!')
			}
		}
		rendered += string(line) + "\n"
	}
	return rendered
}

func activationSummary(genome *neat.Genome) string {
	counts := genome.HiddenActivationCounts()
	types := make([]int, 0, len(counts))
	byValue := map[int]neat.ActivationType{}
	for activation := range counts {
		types = append(types, int(activation))
		byValue[int(activation)] = activation
	}
	sort.Ints(types)

	summary := ""
	for _, value := range types {
		activation := byValue[value]
		if summary != "" {
			summary += " "
		}
		summary += fmt.Sprintf("%s:%d", activation, counts[activation])
	}
	return summary
}

func main() {
	task := benchmarks.NewCubicBoundary(gridResolution)
	fmt.Printf("task: %s (%s), %d labeled points\n", task.Name(), task.Describe(), len(task.Points()))

	_, champion, solved := train(task, 1, maxGenerations, "parabola_population.json")
	accuracy := task.Accuracy(champion)

	fmt.Printf("solved: %v | accuracy %.4f | hidden nodes: %d | distinct activations: %d\n",
		solved, accuracy, len(champion.HiddenIDs()), champion.DistinctHiddenActivations())
	fmt.Printf("activation histogram: %s\n", activationSummary(champion))
	fmt.Println(renderGrid(task, champion, 61, 21))

	data, err := json.MarshalIndent(champion, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("parabola_champion.json", data, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Println("wrote parabola_champion.json and parabola_population.json")
}
