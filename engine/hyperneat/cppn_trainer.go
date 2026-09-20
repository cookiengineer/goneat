package hyperneat

import (
	"errors"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
)

// CppnTrainer performs gradient training of the CPPN weights through the
// substrate it generates.
//
// The substrate weights are produced by the CPPN, so a loss on the substrate
// outputs can be differentiated back to the CPPN parameters with the chain
// rule:
//
//	dLoss/dCPPN = sum_links (dLoss/dWeight) * (dWeight/dCPPNOutput) * (dCPPNOutput/dCPPN)
//
// Each training step rebuilds the substrate from the current CPPN, backprops
// the substrate loss to its link weights, then pushes those gradients through
// the CPPN's own differentiable graph and applies them. This is the HyperNEAT
// counterpart to training a NEAT brain directly.
type CppnTrainer struct {
	cppn      *CPPN
	layout    SubstrateLayout
	options   Options
	trainer   *backprop.Trainer
	loss      backprop.Loss
	cppnGraph *backprop.Graph
	gradient  []float64
}

// NewCppnTrainer creates a CPPN trainer. Defaults are supplied for a nil
// trainer or loss.
func NewCppnTrainer(cppn *CPPN, layout SubstrateLayout, options Options, trainer *backprop.Trainer, loss backprop.Loss) (*CppnTrainer, error) {
	if cppn == nil {
		return nil, errors.New("hyperneat: nil cppn")
	}
	if layout == nil {
		return nil, errors.New("hyperneat: nil layout")
	}
	if trainer == nil {
		weightRange := options.WeightRange * 2
		if weightRange <= 0 {
			weightRange = 6
		}
		trainer = backprop.NewTrainer(backprop.NewSGD(0.05, 0.9), weightRange)
	}
	if loss == nil {
		loss = backprop.MSELoss{}
	}
	return &CppnTrainer{
		cppn:      cppn,
		layout:    layout,
		options:   options,
		trainer:   trainer,
		loss:      loss,
		cppnGraph: backprop.NewGraph(cppn.network),
		gradient:  make([]float64, cppn.OutputCount()),
	}, nil
}

// Learn runs iterations gradient steps and returns the loss of the last step.
func (t *CppnTrainer) Learn(inputs, targets []float64, iterations int) (float64, error) {
	if iterations < 1 {
		iterations = 1
	}
	var last float64
	for i := 0; i < iterations; i++ {
		value, err := t.step(inputs, targets)
		if err != nil {
			return last, err
		}
		last = value
	}
	return last, nil
}

func (t *CppnTrainer) step(inputs, targets []float64) (float64, error) {
	// 1. Rebuild the substrate from the current CPPN weights.
	solver, err := NewSolver(t.layout, t.cppn, t.options)
	if err != nil {
		return 0, err
	}

	// 2. Backprop the substrate loss to its link weights.
	substrateGraph := backprop.NewGraph(solver.Network())
	if _, err := substrateGraph.Forward(inputs); err != nil {
		return 0, err
	}
	lossValue, err := substrateGraph.Backward(t.loss, targets)
	if err != nil {
		return 0, err
	}
	weightGradients := substrateGraph.WeightGradients()

	// 3. Accumulate dLoss/d(cppn weights) across every expressed link.
	t.cppnGraph.ZeroGradients()
	for innovation, weightGradient := range weightGradients {
		if weightGradient == 0 {
			continue
		}
		source, target, ok := solver.LinkPoints(innovation)
		if !ok {
			continue
		}
		outputs, err := t.cppnGraph.Forward([]float64{source.X, source.Y, source.Z, target.X, target.Y, target.Z})
		if err != nil {
			return 0, err
		}
		expressionGradient := expressWeightGradient(outputs, t.options)
		if expressionGradient == 0 {
			continue
		}
		for i := range t.gradient {
			t.gradient[i] = 0
		}
		t.gradient[0] = weightGradient * expressionGradient
		t.cppnGraph.AccumulateOutputGradient(t.gradient)
	}

	// 4. Apply the accumulated gradients to the CPPN genome.
	t.trainer.ApplyGradients(t.cppn.Genome(), t.cppnGraph.WeightGradients())
	return lossValue, nil
}

// Genome returns the CPPN genome being trained.
func (t *CppnTrainer) Genome() *neat.Genome { return t.cppn.Genome() }

// Trainer returns the underlying optimizer wrapper.
func (t *CppnTrainer) Trainer() *backprop.Trainer { return t.trainer }
