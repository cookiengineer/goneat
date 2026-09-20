package neat

import (
	"errors"
	"math"
	"testing"
)

func TestNetworkActivatesBaseGenome(t *testing.T) {
	genome := &Genome{
		Nodes: []NodeGene{
			{ID: 0, Type: NodeInput, Activation: ActivationLinear},
			{ID: 1, Type: NodeInput, Activation: ActivationLinear},
			{ID: 2, Type: NodeBias, Activation: ActivationLinear},
			{ID: 3, Type: NodeOutput, Activation: ActivationSigmoid},
		},
		Connections: []ConnectionGene{
			{Innovation: 1, In: 0, Out: 3, Weight: 1, Enabled: true},
			{Innovation: 2, In: 1, Out: 3, Weight: 1, Enabled: true},
			{Innovation: 3, In: 2, Out: 3, Weight: 0, Enabled: true},
		},
	}

	network, err := NewNetwork(genome)
	if err != nil {
		t.Fatalf("NewNetwork: %v", err)
	}
	outputs := make([]float64, 1)
	if err := network.Activate([]float64{1, 1}, outputs); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	want := 1 / (1 + math.Exp(-2))
	if !almostEqual(outputs[0], want) {
		t.Fatalf("output = %v, want %v", outputs[0], want)
	}
}

func TestNetworkHiddenOrder(t *testing.T) {
	genome := &Genome{
		Nodes: []NodeGene{
			{ID: 0, Type: NodeInput, Activation: ActivationLinear},
			{ID: 1, Type: NodeBias, Activation: ActivationLinear},
			{ID: 2, Type: NodeOutput, Activation: ActivationLinear},
			{ID: 3, Type: NodeHidden, Activation: ActivationLinear},
		},
		Connections: []ConnectionGene{
			{Innovation: 1, In: 0, Out: 3, Weight: 2, Enabled: true},
			{Innovation: 2, In: 3, Out: 2, Weight: 3, Enabled: true},
			{Innovation: 3, In: 1, Out: 2, Weight: 1, Enabled: true},
		},
	}
	network, err := NewNetwork(genome)
	if err != nil {
		t.Fatalf("NewNetwork: %v", err)
	}
	order := network.Order()
	positionOf := func(id int) int {
		for i, candidate := range order {
			if candidate == id {
				return i
			}
		}
		return -1
	}
	if positionOf(3) > positionOf(2) {
		t.Fatalf("hidden node evaluated after output: %v", order)
	}

	outputs := make([]float64, 1)
	if err := network.Activate([]float64{5}, outputs); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	// hidden = 2*5 = 10; output = 3*10 + 1*1 = 31.
	if !almostEqual(outputs[0], 31) {
		t.Fatalf("output = %v, want 31", outputs[0])
	}
}

func TestNetworkMultAggregation(t *testing.T) {
	genome := &Genome{
		Nodes: []NodeGene{
			{ID: 0, Type: NodeInput, Activation: ActivationLinear},
			{ID: 1, Type: NodeInput, Activation: ActivationLinear},
			{ID: 2, Type: NodeOutput, Activation: ActivationLinear},
			{ID: 3, Type: NodeHidden, Activation: ActivationMult},
		},
		Connections: []ConnectionGene{
			{Innovation: 1, In: 0, Out: 3, Weight: 2, Enabled: true},
			{Innovation: 2, In: 1, Out: 3, Weight: 3, Enabled: true},
			{Innovation: 3, In: 3, Out: 2, Weight: 1, Enabled: true},
		},
	}
	network, err := NewNetwork(genome)
	if err != nil {
		t.Fatalf("NewNetwork: %v", err)
	}
	outputs := make([]float64, 1)
	if err := network.Activate([]float64{2, 2}, outputs); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	// hidden = (2*2) * (3*2) = 24.
	if !almostEqual(outputs[0], 24) {
		t.Fatalf("output = %v, want 24", outputs[0])
	}
}

func TestNetworkDetectsCycle(t *testing.T) {
	genome := &Genome{
		Nodes: []NodeGene{
			{ID: 0, Type: NodeInput, Activation: ActivationLinear},
			{ID: 1, Type: NodeOutput, Activation: ActivationLinear},
			{ID: 2, Type: NodeHidden, Activation: ActivationLinear},
			{ID: 3, Type: NodeHidden, Activation: ActivationLinear},
		},
		Connections: []ConnectionGene{
			{Innovation: 1, In: 0, Out: 1, Weight: 1, Enabled: true},
			{Innovation: 2, In: 2, Out: 3, Weight: 1, Enabled: true},
			{Innovation: 3, In: 3, Out: 2, Weight: 1, Enabled: true},
		},
	}
	if _, err := NewNetwork(genome); !errors.Is(err, ErrCyclicNetwork) {
		t.Fatalf("expected ErrCyclicNetwork, got %v", err)
	}
}
