package hyperneat

import (
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
)

// evenCPPNGenome builds a CPPN whose outputs are even in x1 and x2:
//
//	out = |x1| + |x2| + bias
//
// It is mirror-symmetric across the YZ plane, which is exactly the property
// HyperNEAT exploits to transfer a pattern from one side of the body to the
// other.
func evenCPPNGenome(outputs int) *neat.Genome {
	nodes := []neat.NodeGene{
		{ID: 0, Type: neat.NodeInput, Activation: neat.ActivationLinear},
		{ID: 1, Type: neat.NodeInput, Activation: neat.ActivationLinear},
		{ID: 2, Type: neat.NodeInput, Activation: neat.ActivationLinear},
		{ID: 3, Type: neat.NodeInput, Activation: neat.ActivationLinear},
		{ID: 4, Type: neat.NodeInput, Activation: neat.ActivationLinear},
		{ID: 5, Type: neat.NodeInput, Activation: neat.ActivationLinear},
		{ID: 6, Type: neat.NodeBias, Activation: neat.ActivationLinear},
		{ID: 7, Type: neat.NodeHidden, Activation: neat.ActivationAbs},
		{ID: 8, Type: neat.NodeHidden, Activation: neat.ActivationAbs},
	}
	connections := []neat.ConnectionGene{
		{Innovation: 1, In: 0, Out: 7, Weight: 1, Enabled: true},
		{Innovation: 2, In: 3, Out: 8, Weight: 1, Enabled: true},
	}
	innovation := 2
	for output := 0; output < outputs; output++ {
		id := 9 + output
		nodes = append(nodes, neat.NodeGene{ID: id, Type: neat.NodeOutput, Activation: neat.ActivationLinear})
		innovation++
		connections = append(connections, neat.ConnectionGene{Innovation: innovation, In: 7, Out: id, Weight: 1, Enabled: true})
		innovation++
		connections = append(connections, neat.ConnectionGene{Innovation: innovation, In: 8, Out: id, Weight: 1, Enabled: true})
		innovation++
		connections = append(connections, neat.ConnectionGene{Innovation: innovation, In: 6, Out: id, Weight: 0.5, Enabled: true})
	}
	return &neat.Genome{ID: 1, Nodes: nodes, Connections: connections}
}

func TestCPPNIsMirrorEven(t *testing.T) {
	cppn, err := NewCPPN(evenCPPNGenome(1))
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}
	sources := []Point{{X: 0.3, Y: -0.7, Z: 0.1}, {X: -0.2, Y: 1, Z: -0.4}, {X: 0.8, Y: 0.1, Z: 0.6}}
	targets := []Point{{X: -0.5, Y: 0.2, Z: 0.9}, {X: 0.4, Y: -0.3, Z: 0.0}, {X: -0.9, Y: -0.9, Z: 0.3}}

	for _, source := range sources {
		for _, target := range targets {
			original, err := cppn.Query(source, target)
			if err != nil {
				t.Fatalf("Query: %v", err)
			}
			originalValue := original[0]
			mirrored, err := cppn.Query(source.MirrorX(), target.MirrorX())
			if err != nil {
				t.Fatalf("Query mirrored: %v", err)
			}
			if !almostEqual(originalValue, mirrored[0]) {
				t.Fatalf("cppn not mirror even: %v vs %v", originalValue, mirrored[0])
			}
		}
	}
}

// mirroredLayout reflects the inner layout across the YZ plane.
type mirroredLayout struct{ inner SubstrateLayout }

func (m mirroredLayout) NodePosition(index int, nodeType NodeType) (Point, error) {
	point, err := m.inner.NodePosition(index, nodeType)
	return point.MirrorX(), err
}
func (m mirroredLayout) BiasCount() int   { return m.inner.BiasCount() }
func (m mirroredLayout) InputCount() int  { return m.inner.InputCount() }
func (m mirroredLayout) HiddenCount() int { return m.inner.HiddenCount() }
func (m mirroredLayout) OutputCount() int { return m.inner.OutputCount() }

func connectionWeights(genome *neat.Genome) map[[2]int]float64 {
	weights := map[[2]int]float64{}
	for _, connection := range genome.Connections {
		weights[[2]int{connection.In, connection.Out}] = connection.Weight
	}
	return weights
}

func TestSubstrateMirrorGeneralization(t *testing.T) {
	cppn, err := NewCPPN(evenCPPNGenome(1))
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}
	options := DefaultOptions()
	options.LinkThreshold = 0.1
	options.WeightRange = 3

	layout := NewGridSubstrateLayout(1, 2, 0, 1)
	solver, err := NewSolver(layout, cppn, options)
	if err != nil {
		t.Fatalf("NewSolver: %v", err)
	}
	mirroredSolver, err := NewSolver(mirroredLayout{inner: layout}, cppn, options)
	if err != nil {
		t.Fatalf("NewSolver mirrored: %v", err)
	}

	original := connectionWeights(solver.Genome())
	mirrored := connectionWeights(mirroredSolver.Genome())
	if len(original) != len(mirrored) {
		t.Fatalf("link counts differ: %d vs %d", len(original), len(mirrored))
	}
	for key, weight := range original {
		if !almostEqual(weight, mirrored[key]) {
			t.Fatalf("mirrored substrate weight differs at %v: %v vs %v", key, weight, mirrored[key])
		}
	}
}

func TestSubstrateActivates(t *testing.T) {
	cppn, err := NewCPPN(evenCPPNGenome(1))
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}
	options := DefaultOptions()
	options.LinkThreshold = 0.1
	options.OutputActivation = neat.ActivationTanh

	layout := NewGridSubstrateLayout(1, 2, 2, 1)
	solver, err := NewSolver(layout, cppn, options)
	if err != nil {
		t.Fatalf("NewSolver: %v", err)
	}
	if solver.InputCount() != 2 || solver.OutputCount() != 1 {
		t.Fatalf("dims = %d/%d", solver.InputCount(), solver.OutputCount())
	}
	if solver.LinkCount() == 0 {
		t.Fatalf("no links expressed")
	}

	outputs := make([]float64, 1)
	if err := solver.Activate([]float64{0.5, -0.5}, outputs); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	if outputs[0] == 0 {
		t.Logf("note: substrate output is zero for these weights")
	}
}

func TestLinkExpressionOutput(t *testing.T) {
	cppn, err := NewCPPN(evenCPPNGenome(2))
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}
	options := DefaultOptions()
	options.UseLEO = true
	options.WeightRange = 2

	layout := NewGridSubstrateLayout(1, 2, 0, 1)
	solver, err := NewSolver(layout, cppn, options)
	if err != nil {
		t.Fatalf("NewSolver: %v", err)
	}
	if solver.LinkCount() == 0 {
		t.Fatalf("LEO did not express any links")
	}
}

func almostEqual(a, b float64) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= 1e-9
}
