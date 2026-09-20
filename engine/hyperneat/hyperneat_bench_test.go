package hyperneat

import (
	"testing"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
)

// BenchmarkSubstrateTrainStep measures one backprop step on a substrate's own
// connection weights.
func BenchmarkSubstrateTrainStep(b *testing.B) {
	cppn, err := NewCPPN(cppnTrainerGenome(3))
	if err != nil {
		b.Fatalf("NewCPPN: %v", err)
	}
	options := DefaultOptions()
	options.LinkThreshold = 0.0
	options.WeightRange = 1.0
	options.HiddenActivation = neat.ActivationLinear
	options.OutputActivation = neat.ActivationTanh
	layout := NewGridSubstrateLayout(1, 2, 1, 1)
	solver, err := NewSolver(layout, cppn, options)
	if err != nil {
		b.Fatalf("NewSolver: %v", err)
	}

	inputs := cppnTrainerDataset[0].Inputs
	targets := cppnTrainerDataset[0].Targets
	trainer := backprop.NewTrainer(backprop.NewSGD(0.01, 0.9), 4)
	loss := backprop.MSELoss{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := trainer.Learn(solver.Genome(), solver.Network(), inputs, targets, loss); err != nil {
			b.Fatalf("Learn: %v", err)
		}
	}
}

// BenchmarkCppnTrainStep measures one CPPN gradient step, which rebuilds the
// substrate and differentiates through every expressed link.
func BenchmarkCppnTrainStep(b *testing.B) {
	cppn, err := NewCPPN(cppnTrainerGenome(3))
	if err != nil {
		b.Fatalf("NewCPPN: %v", err)
	}
	options := DefaultOptions()
	options.LinkThreshold = 0.0
	options.WeightRange = 1.0
	options.HiddenActivation = neat.ActivationLinear
	options.OutputActivation = neat.ActivationTanh
	layout := NewGridSubstrateLayout(1, 2, 1, 1)

	trainer, err := NewCppnTrainer(cppn, layout, options, backprop.NewTrainer(backprop.NewSGD(0.01, 0.9), 4), backprop.MSELoss{})
	if err != nil {
		b.Fatalf("NewCppnTrainer: %v", err)
	}
	inputs := cppnTrainerDataset[0].Inputs
	targets := cppnTrainerDataset[0].Targets

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := trainer.Learn(inputs, targets, 1); err != nil {
			b.Fatalf("Learn: %v", err)
		}
	}
}
