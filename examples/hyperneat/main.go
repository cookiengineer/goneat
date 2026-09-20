// Command hyperneat demonstrates Evolvable-Substrate HyperNEAT on a small
// retina-style task: a CPPN generates the weights of a substrate that detects
// specific shapes on the left and right half of a retina.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/cookiengineer/goneat/engine/hyperneat"
	"github.com/cookiengineer/goneat/engine/neat"
)

// VisualObject is a shape encoded as two pixels.
type VisualObject struct {
	Name   string
	Pixels [2]float64
}

// RetinaObjects are the shapes the retina can see.
var RetinaObjects = []VisualObject{
	{Name: "none", Pixels: [2]float64{0, 0}},
	{Name: "right", Pixels: [2]float64{0, 1}},
	{Name: "left", Pixels: [2]float64{1, 0}},
}

// DetectionTargets says which shape each output should detect.
var DetectionTargets = [2]string{"left", "right"}

// SubstrateInputs is two pixels for the left object plus two for the right.
const SubstrateInputs = 4

// RetinaFitness evaluates a CPPN by building its substrate and scoring the
// shape-detection task. A perfect substrate scores 1.0.
func RetinaFitness(genome *neat.Genome) (float64, error) {
	cppn, err := hyperneat.NewCPPN(genome)
	if err != nil {
		return 0, nil
	}

	esOptions := hyperneat.DefaultESOptions()
	esOptions.InitialDepth = 2
	esOptions.MaximalDepth = 3
	points, err := hyperneat.DiscoverHiddenPoints(cppn, esOptions)
	if err != nil {
		return 0, nil
	}

	layout := hyperneat.NewMappedEvolvableSubstrateLayout(SubstrateInputs, 2)
	layout.SetHiddenPoints(points)

	solverOptions := hyperneat.DefaultOptions()
	solverOptions.LinkThreshold = 0.0
	solverOptions.OutputActivation = neat.ActivationSigmoid
	solver, err := hyperneat.NewSolver(layout, cppn, solverOptions)
	if err != nil {
		return 0, nil
	}

	inputs := make([]float64, SubstrateInputs)
	outputs := make([]float64, 2)
	errorSum := 0.0
	cases := 0.0
	for _, left := range RetinaObjects {
		for _, right := range RetinaObjects {
			inputs[0], inputs[1] = left.Pixels[0], left.Pixels[1]
			inputs[2], inputs[3] = right.Pixels[0], right.Pixels[1]
			if err := solver.Activate(inputs, outputs); err != nil {
				return 0, nil
			}
			targets := [2]float64{
				boolValue(left.Name == DetectionTargets[0]),
				boolValue(right.Name == DetectionTargets[1]),
			}
			for i := 0; i < 2; i++ {
				predicted := 0.0
				if outputs[i] >= 0.5 {
					predicted = 1
				}
				errorSum += abs(predicted - targets[i])
			}
			cases += 2
		}
	}
	if cases == 0 {
		return 0, nil
	}
	return 1 / (1 + errorSum/cases), nil
}

func boolValue(value bool) float64 {
	if value {
		return 1
	}
	return 0
}

func abs(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}

func main() {
	options := neat.DefaultOptions(hyperneat.CPPNInputs, 1)
	options.Seed = 1
	options.OutputActivation = neat.ActivationTanh
	options.AddLinkRate = 0.12
	options.AddNodeRate = 0.06
	options.PopulationSize = 40

	population, err := neat.NewPopulation(options)
	if err != nil {
		log.Fatal(err)
	}

	best := 0.0
	for generation := 0; generation < 40; generation++ {
		stats, err := population.Epoch(neat.EvaluatorFunc(RetinaFitness), nil)
		if err != nil {
			log.Fatal(err)
		}
		if stats.BestFitness > best {
			best = stats.BestFitness
		}
		// Periodically persist the CPPN population to the current directory.
		if stats.Generation%10 == 0 {
			if data, err := population.Encode(); err == nil {
				if err := os.WriteFile("hyperneat_population.json", data, 0o644); err != nil {
					log.Fatal(err)
				}
			}
		}
		fmt.Printf("gen %3d | species %2d | best %.4f | avg %.4f\n",
			stats.Generation, stats.SpeciesCount, stats.BestFitness, stats.AverageFitness)
	}

	champion := population.BestGenome()
	encoded, err := json.MarshalIndent(champion, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("best fitness %.4f\n", best)
	fmt.Println(string(encoded))
}
