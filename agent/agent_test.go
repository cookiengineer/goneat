package agent

import (
	"math"
	"testing"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/hyperneat"
	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/engine/random"
)

func almostEqual(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

func newGenome() *neat.Genome {
	tracker := neat.NewInnovationTracker(2, 1)
	genome := neat.NewGenome(2, 1, tracker, random.New(3))
	genome.ID = 1
	return genome
}

func TestAgentRewardAccumulatesFitness(t *testing.T) {
	agent := NewAgent("a", nil)
	if err := agent.Reward(2, nil); err != nil {
		t.Fatalf("Reward: %v", err)
	}
	if err := agent.Punish(-3, nil); err != nil {
		t.Fatalf("Punish: %v", err)
	}
	if !almostEqual(agent.Fitness, -1) {
		t.Fatalf("fitness = %v, want -1", agent.Fitness)
	}
}

func TestRewardTriggersLearning(t *testing.T) {
	brain, err := NewNEATBrain(newGenome())
	if err != nil {
		t.Fatalf("NewNEATBrain: %v", err)
	}
	agent := NewAgent("a", brain)
	before := brain.Weights()

	training := &Training{
		Inputs:     []float64{0.5, -0.5},
		Targets:    []float64{1},
		Rate:       0.5,
		Iterations: 1,
	}
	if err := agent.Reward(1, training); err != nil {
		t.Fatalf("Reward: %v", err)
	}

	after := brain.Weights()
	changed := false
	for i := range before {
		if !almostEqual(before[i], after[i]) {
			changed = true
		}
	}
	if !changed {
		t.Fatalf("reward did not change brain weights")
	}
}

func TestCloneIndependence(t *testing.T) {
	brain, err := NewNEATBrain(newGenome())
	if err != nil {
		t.Fatalf("NewNEATBrain: %v", err)
	}
	clone := brain.Clone().(*NEATBrain)

	before := brain.Weights()
	if err := clone.SetWeights(make([]float64, len(before))); err != nil {
		t.Fatalf("SetWeights: %v", err)
	}
	after := brain.Weights()
	for i := range before {
		if !almostEqual(before[i], after[i]) {
			t.Fatalf("clone shares weight storage")
		}
	}
}

func TestEncodeDecodePreservesOutputs(t *testing.T) {
	brain, err := NewNEATBrain(newGenome())
	if err != nil {
		t.Fatalf("NewNEATBrain: %v", err)
	}
	inputs := []float64{0.25, -0.75}
	original := make([]float64, 1)
	if err := brain.Activate(inputs, original); err != nil {
		t.Fatalf("Activate: %v", err)
	}

	data, err := brain.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	restored := &NEATBrain{}
	if err := restored.Decode(data); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	decoded := make([]float64, 1)
	if err := restored.Activate(inputs, decoded); err != nil {
		t.Fatalf("Activate restored: %v", err)
	}
	if !almostEqual(original[0], decoded[0]) {
		t.Fatalf("outputs differ: %v vs %v", original[0], decoded[0])
	}
}

func TestBackpropBrainLearnsXOR(t *testing.T) {
	brain, err := NewBackpropBrain(2, 1, 4, neat.ActivationTanh, backprop.NewSGD(0.5, 0.9), 10, random.New(11))
	if err != nil {
		t.Fatalf("NewBackpropBrain: %v", err)
	}

	patterns := [][3]float64{{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}
	loss := backprop.MSELoss{}
	for iteration := 0; iteration < 3000; iteration++ {
		for _, pattern := range patterns {
			if err := brain.Learn([]float64{pattern[0], pattern[1]}, []float64{pattern[2]}, loss, 0.5, 1); err != nil {
				t.Fatalf("Learn: %v", err)
			}
		}
	}

	outputs := make([]float64, 1)
	for _, pattern := range patterns {
		if err := brain.Activate([]float64{pattern[0], pattern[1]}, outputs); err != nil {
			t.Fatalf("Activate: %v", err)
		}
		predicted := 0.0
		if outputs[0] >= 0.5 {
			predicted = 1
		}
		if predicted != pattern[2] {
			t.Fatalf("pattern %v predicted %v", pattern, outputs[0])
		}
	}
}

func TestHyperNeatBrainTrains(t *testing.T) {
	// Build a small fixed CPPN and substrate.
	cppnOptions := neat.DefaultOptions(hyperneat.CPPNInputs, 1)
	cppnOptions.OutputActivation = neat.ActivationTanh
	tracker := neat.NewInnovationTracker(hyperneat.CPPNInputs, 1)
	cppnGenome := neat.NewSeededGenome(cppnOptions, tracker, random.New(13))
	cppn, err := hyperneat.NewCPPN(cppnGenome)
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}

	substrateOptions := hyperneat.DefaultOptions()
	substrateOptions.LinkThreshold = 0.0
	substrateOptions.OutputActivation = neat.ActivationTanh
	layout := hyperneat.NewGridSubstrateLayout(1, 2, 1, 1)
	solver, err := hyperneat.NewSolver(layout, cppn, substrateOptions)
	if err != nil {
		t.Fatalf("NewSolver: %v", err)
	}

	brain, err := NewHyperNeatBrain(solver, backprop.NewTrainer(backprop.NewSGD(0.1, 0.9), 4), backprop.MSELoss{})
	if err != nil {
		t.Fatalf("NewHyperNeatBrain: %v", err)
	}
	player := NewAgent("substrate", brain)

	before := brain.Weights()
	inputs := []float64{0.5, -0.5}
	outputs := make([]float64, 1)
	if err := player.Activate(inputs, outputs); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	if err := player.Reward(1, &Training{Inputs: inputs, Targets: []float64{1}, Rate: 0.1, Iterations: 80}); err != nil {
		t.Fatalf("Reward: %v", err)
	}
	after := brain.Weights()
	changed := false
	for i := range before {
		if !almostEqual(before[i], after[i]) {
			changed = true
		}
	}
	if !changed {
		t.Fatalf("substrate weights did not change after gradient training")
	}
}
