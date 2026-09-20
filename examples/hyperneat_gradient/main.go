// Command hyperneat_gradient demonstrates gradient training in HyperNEAT on a
// small retina-style detection task.
//
// It shows the two supported paths:
//
//  1. substrate weight training: once a CPPN has produced a substrate network,
//     its connection weights can be refined by ordinary backprop, and
//  2. CPPN training: the loss on the substrate outputs can be differentiated
//     all the way back to the CPPN weights that generated them.
package main

import (
	"fmt"
	"log"

	"github.com/cookiengineer/goneat/agent"
	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/hyperneat"
	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/engine/random"
)

// object is a shape encoded as two pixels.
type object struct {
	name   string
	pixels [2]float64
}

var objects = []object{
	{name: "none", pixels: [2]float64{0, 0}},
	{name: "right", pixels: [2]float64{0, 1}},
	{name: "left", pixels: [2]float64{1, 0}},
}

const substrateInputs = 4

type sample struct {
	inputs  []float64
	targets []float64
}

// dataset builds the left/right shape detection samples.
func dataset() []sample {
	var samples []sample
	for _, left := range objects {
		for _, right := range objects {
			inputs := []float64{left.pixels[0], left.pixels[1], right.pixels[0], right.pixels[1]}
			targets := []float64{boolFloat(left.name == "left"), boolFloat(right.name == "right")}
			samples = append(samples, sample{inputs: inputs, targets: targets})
		}
	}
	return samples
}

func boolFloat(value bool) float64 {
	if value {
		return 1
	}
	return 0
}

func newCPPN(seed int64) (*hyperneat.CPPN, error) {
	options := neat.DefaultOptions(hyperneat.CPPNInputs, 1)
	options.OutputActivation = neat.ActivationTanh
	options.InitialWeightStdDev = 0.6
	tracker := neat.NewInnovationTracker(hyperneat.CPPNInputs, 1)
	return hyperneat.NewCPPN(neat.NewSeededGenome(options, tracker, random.New(seed)))
}

func substrateOptions() hyperneat.Options {
	options := hyperneat.DefaultOptions()
	options.LinkThreshold = 0.0
	options.WeightRange = 1.5
	options.HiddenActivation = neat.ActivationTanh
	options.OutputActivation = neat.ActivationSigmoid
	return options
}

// datasetLoss is the mean squared error of a substrate over the task.
func datasetLoss(solver *hyperneat.Solver, samples []sample) float64 {
	outputs := make([]float64, solver.OutputCount())
	total := 0.0
	for _, s := range samples {
		if err := solver.Activate(s.inputs, outputs); err != nil {
			return total
		}
		value, err := backprop.MSELoss{}.Compute(outputs, s.targets, nil)
		if err == nil {
			total += value
		}
	}
	return total / float64(len(samples))
}

// runSubstrateTraining refines the substrate's own connection weights with
// backprop. It returns the task loss before and after training.
func runSubstrateTraining(seed int64, epochs int) (before, after float64, err error) {
	samples := dataset()
	layout := hyperneat.NewGridSubstrateLayout(1, substrateInputs, 2, 2)
	options := substrateOptions()

	cppn, err := newCPPN(seed)
	if err != nil {
		return 0, 0, err
	}
	solver, err := hyperneat.NewSolver(layout, cppn, options)
	if err != nil {
		return 0, 0, err
	}
	brain, err := agent.NewHyperNeatBrain(solver, backprop.NewTrainer(backprop.NewSGD(0.1, 0.9), 4), backprop.MSELoss{})
	if err != nil {
		return 0, 0, err
	}

	before = datasetLoss(solver, samples)
	for epoch := 0; epoch < epochs; epoch++ {
		for _, s := range samples {
			if err := brain.Learn(s.inputs, s.targets, backprop.MSELoss{}, 0.1, 1); err != nil {
				return before, 0, err
			}
		}
	}
	after = datasetLoss(solver, samples)
	return before, after, nil
}

// runCPPNTraining differentiates the substrate loss back into the CPPN weights.
// It returns the task loss before and after training.
func runCPPNTraining(seed int64, epochs int) (before, after float64, err error) {
	samples := dataset()
	layout := hyperneat.NewGridSubstrateLayout(1, substrateInputs, 2, 2)
	options := substrateOptions()

	cppn, err := newCPPN(seed)
	if err != nil {
		return 0, 0, err
	}
	trainer, err := hyperneat.NewCppnTrainer(cppn, layout, options, backprop.NewTrainer(backprop.NewSGD(0.02, 0.9), 4), backprop.MSELoss{})
	if err != nil {
		return 0, 0, err
	}
	solver, err := hyperneat.NewSolver(layout, cppn, options)
	if err != nil {
		return 0, 0, err
	}

	before = datasetLoss(solver, samples)
	for epoch := 0; epoch < epochs; epoch++ {
		for _, s := range samples {
			if _, err := trainer.Learn(s.inputs, s.targets, 1); err != nil {
				return before, 0, err
			}
		}
	}
	// Rebuild the substrate from the refined CPPN before measuring.
	solver, err = hyperneat.NewSolver(layout, cppn, options)
	if err != nil {
		return before, 0, err
	}
	after = datasetLoss(solver, samples)
	return before, after, nil
}

func main() {
	before, after, err := runSubstrateTraining(1, 200)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("substrate training: loss %.6f -> %.6f\n", before, after)

	before, after, err = runCPPNTraining(2, 150)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("cppn training:      loss %.6f -> %.6f\n", before, after)
}
