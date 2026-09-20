package neat

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/cookiengineer/goneat/engine/random"
)

func newTestRand(seed int64) *random.Rand {
	return random.New(seed)
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= 1e-9
}

func TestActivationApplyAndDerivative(t *testing.T) {
	if got := ActivationSigmoid.Apply(0); !almostEqual(got, 0.5) {
		t.Fatalf("sigmoid(0) = %v", got)
	}
	if got := ActivationSigmoid.Derivative(0); !almostEqual(got, 0.25) {
		t.Fatalf("sigmoid'(0) = %v", got)
	}
	if got := ActivationTanh.Apply(0); got != 0 {
		t.Fatalf("tanh(0) = %v", got)
	}
	if got := ActivationTanh.Derivative(0); got != 1 {
		t.Fatalf("tanh'(0) = %v", got)
	}
	if got := ActivationReLU.Apply(-3); got != 0 {
		t.Fatalf("relu(-3) = %v", got)
	}
	if got := ActivationReLU.Apply(3); got != 3 {
		t.Fatalf("relu(3) = %v", got)
	}
	if got := ActivationAbs.Apply(-2); got != 2 {
		t.Fatalf("abs(-2) = %v", got)
	}
}

func TestActivationTextRoundTrip(t *testing.T) {
	original := ActivationGaussian
	text, err := original.MarshalText()
	if err != nil {
		t.Fatalf("MarshalText: %v", err)
	}
	var decoded ActivationType
	if err := decoded.UnmarshalText(text); err != nil {
		t.Fatalf("UnmarshalText: %v", err)
	}
	if decoded != original {
		t.Fatalf("round trip = %v, want %v", decoded, original)
	}
}

func TestActivationAggregation(t *testing.T) {
	if !ActivationMult.IsAggregation() || !ActivationAdd.IsAggregation() {
		t.Fatalf("mult/add should be aggregations")
	}
	if ActivationSigmoid.IsAggregation() {
		t.Fatalf("sigmoid should not be an aggregation")
	}
}

func TestGenomeBaseTopology(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(1))

	if len(genome.InputIDs()) != 2 {
		t.Fatalf("inputs = %v", genome.InputIDs())
	}
	if len(genome.OutputIDs()) != 1 {
		t.Fatalf("outputs = %v", genome.OutputIDs())
	}
	if _, ok := genome.BiasID(); !ok {
		t.Fatalf("expected a bias node")
	}
	if len(genome.Connections) != (2+1)*1 {
		t.Fatalf("connections = %d, want 3", len(genome.Connections))
	}
	if len(genome.HiddenIDs()) != 0 {
		t.Fatalf("unexpected hidden nodes: %v", genome.HiddenIDs())
	}
}

func TestGenomeCloneIsIndependent(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(2))
	clone := genome.Clone()

	clone.Connections[0].Weight = 999
	clone.Nodes[0].Activation = ActivationTanh
	if genome.Connections[0].Weight == 999 {
		t.Fatalf("clone shares connection storage")
	}
	if genome.Nodes[0].Activation == ActivationTanh {
		t.Fatalf("clone shares node storage")
	}
}

func TestGenomeJSONRoundTrip(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(3))
	genome.ID = 7
	genome.Fitness = 1.5
	genome.Generation = 4

	encoded, err := json.Marshal(genome)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var decoded Genome
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if decoded.ID != 7 || !almostEqual(decoded.Fitness, 1.5) || decoded.Generation != 4 {
		t.Fatalf("decoded metadata = %+v", decoded)
	}
	if len(decoded.Nodes) != len(genome.Nodes) || len(decoded.Connections) != len(genome.Connections) {
		t.Fatalf("decoded sizes differ: %d/%d nodes, %d/%d connections",
			len(decoded.Nodes), len(genome.Nodes), len(decoded.Connections), len(genome.Connections))
	}
	if decoded.Nodes[0].Type != NodeInput || decoded.Nodes[len(decoded.Nodes)-1].Type != NodeOutput {
		t.Fatalf("decoded node types = %+v", decoded.Nodes)
	}
	if decoded.Connections[0].Innovation != genome.Connections[0].Innovation {
		t.Fatalf("innovation mismatch")
	}
}
