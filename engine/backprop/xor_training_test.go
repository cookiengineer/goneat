package backprop

import (
	"math"
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/engine/random"
)

// This file is the canonical proof that gradient training actually works. It
// trains small feed-forward networks on XOR, which is not linearly separable
// and therefore cannot be learned without a hidden layer. Because the
// generator and the whole training loop are deterministic, the tests also act
// as regression guards: a change that worsens training will change the
// iterations-to-solve and final loss and trip these bounds.

// xorTrainingPatterns is the XOR truth table: {input0, input1, target}.
var xorTrainingPatterns = [][3]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

// xorConfig fully describes one reproducible XOR training run.
type xorConfig struct {
	Name         string
	Loss         Loss
	NewOptimizer func() Optimizer
	Hidden       int
	Seed         int64
	MaxPasses    int
	// StopOnSolve stops as soon as all four patterns are classified correctly,
	// which measures time-to-solution. When false it runs the full budget,
	// which measures convergence quality for a fixed cost.
	StopOnSolve bool
}

// xorResult is the outcome of a training run.
type xorResult struct {
	Iterations  int
	InitialLoss float64
	FinalLoss   float64
	Accuracy    float64
	Solved      bool
	Weights     []float64
}

// xorHiddenGenome builds 2 inputs -> hidden tanh neurons -> 1 sigmoid output
// plus a bias connection to the output. Weight order is deterministic.
func xorHiddenGenome(hidden int, weights []float64) *neat.Genome {
	nodes := []neat.NodeGene{
		{ID: 0, Type: neat.NodeInput, Activation: neat.ActivationLinear},
		{ID: 1, Type: neat.NodeInput, Activation: neat.ActivationLinear},
		{ID: 2, Type: neat.NodeBias, Activation: neat.ActivationLinear},
	}
	hiddenIDs := make([]int, hidden)
	for h := 0; h < hidden; h++ {
		id := 3 + h
		hiddenIDs[h] = id
		nodes = append(nodes, neat.NodeGene{ID: id, Type: neat.NodeHidden, Activation: neat.ActivationTanh})
	}
	outputID := 3 + hidden
	nodes = append(nodes, neat.NodeGene{ID: outputID, Type: neat.NodeOutput, Activation: neat.ActivationSigmoid})

	var connections []neat.ConnectionGene
	innovation, cursor := 0, 0
	add := func(in, out int) {
		innovation++
		connections = append(connections, neat.ConnectionGene{
			Innovation: innovation,
			In:         in,
			Out:        out,
			Weight:     weights[cursor],
			Enabled:    true,
		})
		cursor++
	}
	for _, id := range hiddenIDs {
		add(0, id)
		add(1, id)
		add(2, id)
	}
	for _, id := range hiddenIDs {
		add(id, outputID)
	}
	add(2, outputID)

	return &neat.Genome{ID: 1, Nodes: nodes, Connections: connections}
}

// xorHiddenWeightCount returns the number of weights produced by
// xorHiddenGenome.
func xorHiddenWeightCount(hidden int) int { return hidden*4 + 1 }

// trainXOR runs the configured training and returns the measured outcome.
func trainXOR(config xorConfig) xorResult {
	generator := random.New(config.Seed)
	weights := make([]float64, xorHiddenWeightCount(config.Hidden))
	for i := range weights {
		weights[i] = generator.NormFloat64()
	}
	genome := xorHiddenGenome(config.Hidden, weights)
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return xorResult{Iterations: config.MaxPasses, FinalLoss: math.Inf(1)}
	}
	trainer := NewTrainer(config.NewOptimizer(), 10)
	outputs := make([]float64, network.OutputCount())

	initialLoss, _ := evaluateXOR(network, config.Loss, outputs)
	result := xorResult{InitialLoss: initialLoss, Iterations: config.MaxPasses}

	for pass := 0; pass < config.MaxPasses; pass++ {
		for _, pattern := range xorTrainingPatterns {
			if _, err := trainer.Learn(genome, network, []float64{pattern[0], pattern[1]}, []float64{pattern[2]}, config.Loss); err != nil {
				return xorResult{Iterations: config.MaxPasses, FinalLoss: math.Inf(1)}
			}
		}
		if config.StopOnSolve {
			if _, accuracy := evaluateXOR(network, config.Loss, outputs); accuracy == 1 {
				result.Iterations = pass + 1
				break
			}
		}
	}

	result.FinalLoss, result.Accuracy = evaluateXOR(network, config.Loss, outputs)
	result.Solved = result.Accuracy == 1
	result.Weights = genome.WeightVector()
	return result
}

// evaluateXOR returns the mean loss and classification accuracy over the XOR
// truth table.
func evaluateXOR(network *neat.Network, loss Loss, outputs []float64) (float64, float64) {
	totalLoss := 0.0
	correct := 0
	for _, pattern := range xorTrainingPatterns {
		if err := network.Activate([]float64{pattern[0], pattern[1]}, outputs); err != nil {
			return math.Inf(1), 0
		}
		value, err := loss.Compute(outputs, []float64{pattern[2]}, nil)
		if err == nil {
			totalLoss += value
		}
		if (outputs[0] >= 0.5) == (pattern[2] == 1) {
			correct++
		}
	}
	return totalLoss / float64(len(xorTrainingPatterns)), float64(correct) / float64(len(xorTrainingPatterns))
}

// convergingConfigs are loss/optimizer pairings that solve XOR reliably with a
// hidden layer. They are the backbone of the correctness tests and the
// benchmark comparison.
func convergingConfigs() []xorConfig {
	return []xorConfig{
		{
			Name:         "mse-sgd",
			Loss:         MSELoss{},
			NewOptimizer: func() Optimizer { return NewSGD(0.5, 0.9) },
		},
		{
			Name:         "cross-entropy-rmsprop",
			Loss:         CrossEntropyLoss{},
			NewOptimizer: func() Optimizer { return NewRMSProp(0.1, 0.9, 1e-8, 0, 5) },
		},
	}
}

func TestXORTrainingConvergesAcrossSeeds(t *testing.T) {
	for _, base := range convergingConfigs() {
		base := base
		t.Run(base.Name, func(t *testing.T) {
			for seed := int64(1); seed <= 10; seed++ {
				config := base
				config.Hidden = 4
				config.Seed = seed
				config.MaxPasses = 2000
				config.StopOnSolve = true

				result := trainXOR(config)
				if !result.Solved {
					t.Fatalf("seed %d did not solve XOR: accuracy %.2f, loss %.4f, iterations %d",
						seed, result.Accuracy, result.FinalLoss, result.Iterations)
				}
				if result.FinalLoss >= result.InitialLoss {
					t.Fatalf("seed %d loss did not decrease: initial %.4f, final %.4f",
						seed, result.InitialLoss, result.FinalLoss)
				}
			}
		})
	}
}

func TestXORTrainingIsDeterministic(t *testing.T) {
	config := xorConfig{
		Name:         "mse-sgd",
		Loss:         MSELoss{},
		NewOptimizer: func() Optimizer { return NewSGD(0.5, 0.9) },
		Hidden:       4,
		Seed:         7,
		MaxPasses:    2000,
		StopOnSolve:  true,
	}
	first := trainXOR(config)
	second := trainXOR(config)

	if first.Iterations != second.Iterations {
		t.Fatalf("iterations differ: %d vs %d", first.Iterations, second.Iterations)
	}
	if first.FinalLoss != second.FinalLoss {
		t.Fatalf("final loss differs: %v vs %v", first.FinalLoss, second.FinalLoss)
	}
	if len(first.Weights) != len(second.Weights) {
		t.Fatalf("weight counts differ")
	}
	for i := range first.Weights {
		if first.Weights[i] != second.Weights[i] {
			t.Fatalf("weight %d differs: %v vs %v", i, first.Weights[i], second.Weights[i])
		}
	}
}

// TestXORTrainingRegressionBound pins the reproducible outcome for a canonical
// configuration. If a future change makes training slower or worse, these
// bounds fail, which is exactly the signal we want from the test suite.
func TestXORTrainingRegressionBound(t *testing.T) {
	config := xorConfig{
		Name:         "mse-sgd",
		Loss:         MSELoss{},
		NewOptimizer: func() Optimizer { return NewSGD(0.5, 0.9) },
		Hidden:       4,
		Seed:         7,
		MaxPasses:    2000,
		StopOnSolve:  true,
	}
	result := trainXOR(config)
	if !result.Solved {
		t.Fatalf("canonical config failed to solve XOR")
	}
	if result.Iterations > 200 {
		t.Fatalf("canonical config solved in %d passes, expected <= 200", result.Iterations)
	}
	if result.FinalLoss > 0.15 {
		t.Fatalf("canonical config final loss %.4f exceeded regression bound 0.15", result.FinalLoss)
	}
}

// TestXORLossFunctionsReduceLoss checks, for a fixed budget, that both losses
// move in the right direction. It is the equal-cost counterpart to the
// time-to-solution test above.
func TestXORLossFunctionsReduceLoss(t *testing.T) {
	for _, base := range convergingConfigs() {
		base := base
		t.Run(base.Name, func(t *testing.T) {
			config := base
			config.Hidden = 4
			config.Seed = 7
			config.MaxPasses = 200
			config.StopOnSolve = false

			result := trainXOR(config)
			if result.FinalLoss >= result.InitialLoss {
				t.Fatalf("loss did not decrease over %d passes: initial %.4f, final %.4f",
					config.MaxPasses, result.InitialLoss, result.FinalLoss)
			}
			if result.Accuracy < 0.75 {
				t.Fatalf("accuracy %.2f too low after %d passes", result.Accuracy, config.MaxPasses)
			}
		})
	}
}
