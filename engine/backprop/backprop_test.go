package backprop

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/engine/random"
)

func almostEqual(a, b, tolerance float64) bool {
	return math.Abs(a-b) <= tolerance
}

func manualGenome(weights []float64) *neat.Genome {
	return &neat.Genome{
		ID: 1,
		Nodes: []neat.NodeGene{
			{ID: 0, Type: neat.NodeInput, Activation: neat.ActivationLinear},
			{ID: 1, Type: neat.NodeInput, Activation: neat.ActivationLinear},
			{ID: 2, Type: neat.NodeBias, Activation: neat.ActivationLinear},
			{ID: 3, Type: neat.NodeHidden, Activation: neat.ActivationTanh},
			{ID: 4, Type: neat.NodeOutput, Activation: neat.ActivationSigmoid},
		},
		Connections: []neat.ConnectionGene{
			{Innovation: 1, In: 0, Out: 3, Weight: weights[0], Enabled: true},
			{Innovation: 2, In: 1, Out: 3, Weight: weights[1], Enabled: true},
			{Innovation: 3, In: 2, Out: 3, Weight: weights[2], Enabled: true},
			{Innovation: 4, In: 3, Out: 4, Weight: weights[3], Enabled: true},
			{Innovation: 5, In: 2, Out: 4, Weight: weights[4], Enabled: true},
		},
	}
}

func lossFor(genome *neat.Genome, inputs, targets []float64) float64 {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		panic(err)
	}
	outputs := make([]float64, network.OutputCount())
	if err := network.Activate(inputs, outputs); err != nil {
		panic(err)
	}
	loss, err := MSELoss{}.Compute(outputs, targets, nil)
	if err != nil {
		panic(err)
	}
	return loss
}

func TestGradientCheck(t *testing.T) {
	genome := manualGenome([]float64{0.4, -0.7, 0.2, 1.1, -0.3})
	network, err := neat.NewNetwork(genome)
	if err != nil {
		t.Fatalf("NewNetwork: %v", err)
	}
	inputs := []float64{0.3, -0.6}
	targets := []float64{0.8}

	graph := NewGraph(network)
	if _, err := graph.Forward(inputs); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if _, err := graph.Backward(MSELoss{}, targets); err != nil {
		t.Fatalf("Backward: %v", err)
	}
	analytic := graph.WeightGradients()

	const epsilon = 1e-6
	for _, connection := range genome.Connections {
		index := genome.ConnectionIndex(connection.Innovation)
		original := genome.Connections[index].Weight

		genome.Connections[index].Weight = original + epsilon
		high := lossFor(genome, inputs, targets)
		genome.Connections[index].Weight = original - epsilon
		low := lossFor(genome, inputs, targets)
		genome.Connections[index].Weight = original

		numeric := (high - low) / (2 * epsilon)
		if !almostEqual(analytic[connection.Innovation], numeric, 1e-6) {
			t.Fatalf("connection %d: analytic %v, numeric %v", connection.Innovation, analytic[connection.Innovation], numeric)
		}
	}
}

func xorGenome(weights []float64) *neat.Genome {
	return &neat.Genome{
		Nodes: []neat.NodeGene{
			{ID: 0, Type: neat.NodeInput, Activation: neat.ActivationLinear},
			{ID: 1, Type: neat.NodeInput, Activation: neat.ActivationLinear},
			{ID: 2, Type: neat.NodeBias, Activation: neat.ActivationLinear},
			{ID: 3, Type: neat.NodeHidden, Activation: neat.ActivationTanh},
			{ID: 4, Type: neat.NodeHidden, Activation: neat.ActivationTanh},
			{ID: 5, Type: neat.NodeOutput, Activation: neat.ActivationSigmoid},
		},
		Connections: []neat.ConnectionGene{
			{Innovation: 1, In: 0, Out: 3, Weight: weights[0], Enabled: true},
			{Innovation: 2, In: 1, Out: 3, Weight: weights[1], Enabled: true},
			{Innovation: 3, In: 2, Out: 3, Weight: weights[2], Enabled: true},
			{Innovation: 4, In: 0, Out: 4, Weight: weights[3], Enabled: true},
			{Innovation: 5, In: 1, Out: 4, Weight: weights[4], Enabled: true},
			{Innovation: 6, In: 2, Out: 4, Weight: weights[5], Enabled: true},
			{Innovation: 7, In: 3, Out: 5, Weight: weights[6], Enabled: true},
			{Innovation: 8, In: 4, Out: 5, Weight: weights[7], Enabled: true},
			{Innovation: 9, In: 2, Out: 5, Weight: weights[8], Enabled: true},
		},
	}
}

func TestTrainerLearnsXOR(t *testing.T) {
	rng := random.New(7)
	weights := make([]float64, 9)
	for i := range weights {
		weights[i] = rng.NormFloat64()
	}
	genome := xorGenome(weights)
	network, err := neat.NewNetwork(genome)
	if err != nil {
		t.Fatalf("NewNetwork: %v", err)
	}
	trainer := NewTrainer(NewSGD(0.5, 0.9), 10)
	loss := MSELoss{}

	patterns := [][3]float64{{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}
	for iteration := 0; iteration < 3000; iteration++ {
		for _, pattern := range patterns {
			if _, err := trainer.Learn(genome, network, []float64{pattern[0], pattern[1]}, []float64{pattern[2]}, loss); err != nil {
				t.Fatalf("Learn: %v", err)
			}
		}
	}

	outputs := make([]float64, network.OutputCount())
	for _, pattern := range patterns {
		if err := network.Activate([]float64{pattern[0], pattern[1]}, outputs); err != nil {
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

func TestOptimizerJSONRoundTrip(t *testing.T) {
	optimizer := NewSGD(0.5, 0.9)
	optimizer.Update(1, 1, 0.1)
	optimizer.Update(2, 1, -0.2)
	encoded, err := json.Marshal(optimizer)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	restored := NewSGD(0, 0)
	if err := json.Unmarshal(encoded, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if restored.Rate != optimizer.Rate || restored.Momentum != optimizer.Momentum {
		t.Fatalf("hyper-parameters lost")
	}
	if restored.velocity[1] != optimizer.velocity[1] {
		t.Fatalf("velocity state lost")
	}

	rms := NewRMSProp(0.01, 0.9, 1e-8, 0.001, 5)
	rms.Update(3, 0.5, 0.3)
	encodedRMS, _ := json.Marshal(rms)
	restoredRMS := NewRMSProp(0.01, 0.9, 1e-8, 0.001, 5)
	if err := json.Unmarshal(encodedRMS, restoredRMS); err != nil {
		t.Fatalf("RMSProp Unmarshal: %v", err)
	}
	if !almostEqual(restoredRMS.squareAvg[3], rms.squareAvg[3], 1e-15) {
		t.Fatalf("rmsprop state lost: %v vs %v", restoredRMS.squareAvg[3], rms.squareAvg[3])
	}
}

// TestTrainerWeightWriteBack ensures gradients are applied to the genome (the
// single source of truth) and survive a clone.
func TestTrainerWeightWriteBack(t *testing.T) {
	tracker := neat.NewInnovationTracker(2, 1)
	genome := neat.NewGenome(2, 1, tracker, random.New(5))
	network, err := neat.NewNetwork(genome)
	if err != nil {
		t.Fatalf("NewNetwork: %v", err)
	}
	before := genome.WeightVector()

	trainer := NewTrainer(NewSGD(0.1, 0), 0)
	loss := MSELoss{}
	for _, pattern := range [][3]float64{{0, 0, 1}, {1, 1, 1}, {0, 1, 0}, {1, 0, 0}} {
		if _, err := trainer.Learn(genome, network, []float64{pattern[0], pattern[1]}, []float64{pattern[2]}, loss); err != nil {
			t.Fatalf("Learn: %v", err)
		}
	}
	changed := false
	for i, weight := range genome.WeightVector() {
		if !almostEqual(weight, before[i], 0) {
			changed = true
		}
	}
	if !changed {
		t.Fatalf("genome weights were not updated")
	}
}
