package hyperneat

import (
	"math"
	"testing"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/engine/random"
)

// cppnTrainerDataset is a tiny linear regression task over four inputs.
var cppnTrainerDataset = []struct {
	Inputs  []float64
	Targets []float64
}{
	{[]float64{-1, -1}, []float64{-1}},
	{[]float64{-1, 1}, []float64{0}},
	{[]float64{1, -1}, []float64{0}},
	{[]float64{1, 1}, []float64{1}},
}

func cppnTrainerGenome(seed int64) *neat.Genome {
	options := neat.DefaultOptions(CPPNInputs, 1)
	options.OutputActivation = neat.ActivationTanh
	options.InitialWeightStdDev = 0.5
	tracker := neat.NewInnovationTracker(CPPNInputs, 1)
	return neat.NewSeededGenome(options, tracker, random.New(seed))
}

func cppnTrainerLoss(t *testing.T, trainer *CppnTrainer, options Options, layout SubstrateLayout) float64 {
	t.Helper()
	total := 0.0
	for _, sample := range cppnTrainerDataset {
		total += cppnSampleLoss(t, trainer, options, layout, sample.Inputs, sample.Targets)
	}
	return total / float64(len(cppnTrainerDataset))
}

func cppnSampleLoss(t *testing.T, trainer *CppnTrainer, options Options, layout SubstrateLayout, inputs, targets []float64) float64 {
	t.Helper()
	solver, err := NewSolver(layout, trainer.cppn, options)
	if err != nil {
		t.Fatalf("NewSolver: %v", err)
	}
	outputs := make([]float64, solver.OutputCount())
	if err := solver.Activate(inputs, outputs); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	value, err := backprop.MSELoss{}.Compute(outputs, targets, nil)
	if err != nil {
		t.Fatalf("Compute: %v", err)
	}
	return value
}

func TestCppnTrainerReducesLoss(t *testing.T) {
	cppn, err := NewCPPN(cppnTrainerGenome(3))
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}
	options := DefaultOptions()
	options.LinkThreshold = 0.0
	options.WeightRange = 1.0
	options.HiddenActivation = neat.ActivationLinear
	options.OutputActivation = neat.ActivationTanh

	layout := NewGridSubstrateLayout(1, 2, 1, 1)
	trainer, err := NewCppnTrainer(cppn, layout, options, backprop.NewTrainer(backprop.NewSGD(0.01, 0.9), 4), backprop.MSELoss{})
	if err != nil {
		t.Fatalf("NewCppnTrainer: %v", err)
	}

	initial := cppnTrainerLoss(t, trainer, options, layout)
	for epoch := 0; epoch < 1500; epoch++ {
		for _, sample := range cppnTrainerDataset {
			if _, err := trainer.Learn(sample.Inputs, sample.Targets, 1); err != nil {
				t.Fatalf("Learn: %v", err)
			}
		}
	}
	final := cppnTrainerLoss(t, trainer, options, layout)

	if final >= initial {
		t.Fatalf("cppn training did not reduce loss: initial %.6f, final %.6f", initial, final)
	}
	if final > initial*0.5 {
		t.Fatalf("cppn training reduced loss insufficiently: initial %.6f, final %.6f", initial, final)
	}
}

// TestCppnTrainerGradientDirection checks the CPPN gradient against a small
// numerical perturbation of a single CPPN weight, validating the chain rule
// through the substrate.
func TestCppnTrainerGradientDirection(t *testing.T) {
	cppn, err := NewCPPN(cppnTrainerGenome(11))
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}
	options := DefaultOptions()
	options.LinkThreshold = 0.0
	options.WeightRange = 1.0
	options.HiddenActivation = neat.ActivationLinear
	options.OutputActivation = neat.ActivationTanh

	layout := NewGridSubstrateLayout(1, 2, 1, 1)
	trainer, err := NewCppnTrainer(cppn, layout, options, backprop.NewTrainer(backprop.NewSGD(0, 0), 0), backprop.MSELoss{})
	if err != nil {
		t.Fatalf("NewCppnTrainer: %v", err)
	}

	// Run one step to populate CPPN weight gradients. The optimizer has rate 0
	// so the weights are unchanged and the analytic gradients match the
	// current genome.
	sample := cppnTrainerDataset[3]
	if _, err := trainer.Learn(sample.Inputs, sample.Targets, 1); err != nil {
		t.Fatalf("Learn: %v", err)
	}
	gradients := trainer.cppnGraph.WeightGradients()

	const epsilon = 1e-5
	for _, connection := range cppn.Genome().Connections {
		gradient := gradients[connection.Innovation]
		if math.Abs(gradient) < 1e-9 {
			continue
		}
		index := cppn.Genome().ConnectionIndex(connection.Innovation)
		original := cppn.Genome().Connections[index].Weight

		cppn.Genome().Connections[index].Weight = original + epsilon
		high := cppnSampleLoss(t, trainer, options, layout, sample.Inputs, sample.Targets)
		cppn.Genome().Connections[index].Weight = original - epsilon
		low := cppnSampleLoss(t, trainer, options, layout, sample.Inputs, sample.Targets)
		cppn.Genome().Connections[index].Weight = original

		numeric := (high - low) / (2 * epsilon)
		if math.Abs(gradient-numeric) > 1e-4 {
			t.Fatalf("cppn gradient mismatch for innovation %d: analytic %v, numeric %v",
				connection.Innovation, gradient, numeric)
		}
	}
}
