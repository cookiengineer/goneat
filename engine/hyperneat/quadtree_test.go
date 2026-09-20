package hyperneat

import (
	"testing"

	"github.com/cookiengineer/goneat/engine/neat"
)

func TestDiscoverHiddenPoints(t *testing.T) {
	cppn, err := NewCPPN(evenCPPNGenome(1))
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}
	options := DefaultESOptions()
	points, err := DiscoverHiddenPoints(cppn, options)
	if err != nil {
		t.Fatalf("DiscoverHiddenPoints: %v", err)
	}
	if len(points) == 0 {
		t.Fatalf("no hidden points discovered")
	}
	for _, point := range points {
		if point.X < -1.001 || point.X > 1.001 || point.Y < -1.001 || point.Y > 1.001 {
			t.Fatalf("discovered point outside hypercube: %+v", point)
		}
	}
}

func TestEvolvableSubstrateBuildsSolver(t *testing.T) {
	cppn, err := NewCPPN(evenCPPNGenome(1))
	if err != nil {
		t.Fatalf("NewCPPN: %v", err)
	}
	points, err := DiscoverHiddenPoints(cppn, DefaultESOptions())
	if err != nil {
		t.Fatalf("DiscoverHiddenPoints: %v", err)
	}

	layout := NewMappedEvolvableSubstrateLayout(2, 2)
	layout.SetHiddenPoints(points)

	options := DefaultOptions()
	options.LinkThreshold = 0.0
	options.OutputActivation = neat.ActivationTanh
	solver, err := NewSolver(layout, cppn, options)
	if err != nil {
		t.Fatalf("NewSolver: %v", err)
	}
	if solver.InputCount() != 2 || solver.OutputCount() != 2 {
		t.Fatalf("dims = %d/%d", solver.InputCount(), solver.OutputCount())
	}
	if solver.LinkCount() == 0 {
		t.Fatalf("no links expressed for evolvable substrate")
	}

	outputs := make([]float64, 2)
	if err := solver.Activate([]float64{0.4, -0.4}, outputs); err != nil {
		t.Fatalf("Activate: %v", err)
	}
}
