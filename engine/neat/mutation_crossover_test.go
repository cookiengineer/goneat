package neat

import "testing"

func TestAddNodeSplitsConnection(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(10))
	options := DefaultOptions(2, 1)

	before := len(genome.Connections)
	for i := 0; i < 50; i++ {
		if AddNode(genome, tracker, options, newTestRand(int64(i))) {
			break
		}
	}

	if len(genome.HiddenIDs()) != 1 {
		t.Fatalf("expected one hidden node, got %v", genome.HiddenIDs())
	}
	if len(genome.Connections) != before+2 {
		t.Fatalf("connections = %d, want %d", len(genome.Connections), before+2)
	}
	if _, err := NewNetwork(genome); err != nil {
		t.Fatalf("network after AddNode: %v", err)
	}
}

func TestAddLinkAvoidsCycles(t *testing.T) {
	tracker := NewInnovationTracker(3, 1)
	options := DefaultOptions(3, 1)
	options.InitialConnectionRate = 0.4
	genome := NewSeededGenome(options, tracker, newTestRand(11))

	added := 0
	for i := 0; i < 200; i++ {
		if AddLink(genome, tracker, options, newTestRand(int64(100+i))) {
			added++
		}
		if _, err := NewNetwork(genome); err != nil {
			t.Fatalf("cycle introduced after %d adds: %v", added, err)
		}
	}
	if added == 0 {
		t.Fatalf("expected at least one link to be added")
	}
}

func TestMutateWeightsChangesValues(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(12))
	options := DefaultOptions(2, 1)
	options.WeightPerturbRate = 1.0
	options.WeightResetRate = 0

	original := genome.WeightVector()
	for i := 0; i < 10; i++ {
		MutateWeights(genome, options, newTestRand(int64(i)))
	}
	changed := false
	for i, weight := range genome.WeightVector() {
		if !almostEqual(weight, original[i]) {
			changed = true
		}
	}
	if !changed {
		t.Fatalf("weights did not change")
	}
}

func TestToggleEnable(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(13))
	before := genome.Connections[0].Enabled
	ToggleEnable(genome, newTestRand(1))
	if genome.Connections[0].Enabled == before {
		// The toggle picks a random index; run until index 0 is selected.
		for i := 0; i < 100 && genome.Connections[0].Enabled == before; i++ {
			ToggleEnable(genome, newTestRand(int64(i)))
		}
	}
	count := 0
	for _, connection := range genome.Connections {
		if connection.Enabled {
			count++
		}
	}
	if count == len(genome.Connections) {
		t.Fatalf("expected at least one disabled connection")
	}
}

func TestCrossoverInheritsFitterDisjoint(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	fitter := NewGenome(2, 1, tracker, newTestRand(20))
	other := NewGenome(2, 1, tracker, newTestRand(21))

	// Disable one shared gene; child should keep it disabled when both are.
	fitter.Connections[0].Enabled = false
	other.Connections[0].Enabled = false

	// Give the fitter an extra gene the other lacks.
	extraInnovation := 500
	fitter.Connections = append(fitter.Connections, ConnectionGene{
		Innovation: extraInnovation,
		In:         2,
		Out:        3,
		Weight:     0.5,
		Enabled:    true,
	})

	fitter.Fitness = 10
	other.Fitness = 1

	child := Crossover(fitter, other, newTestRand(23))
	if child.ID != -1 {
		t.Fatalf("child ID = %d, want -1", child.ID)
	}
	if len(child.Parents) != 2 || child.Parents[0] != fitter.ID || child.Parents[1] != other.ID {
		t.Fatalf("child parents = %v", child.Parents)
	}
	if child.ConnectionIndex(extraInnovation) < 0 {
		t.Fatalf("child did not inherit fitter's disjoint gene %d", extraInnovation)
	}
	if gene := child.Connections[child.ConnectionIndex(fitter.Connections[0].Innovation)]; gene.Enabled {
		t.Fatalf("gene disabled in both parents should stay disabled")
	}
	if len(child.InputIDs()) != 2 || len(child.OutputIDs()) != 1 {
		t.Fatalf("child topology inputs/outputs = %v/%v", child.InputIDs(), child.OutputIDs())
	}
	if _, err := NewNetwork(child); err != nil {
		t.Fatalf("child network: %v", err)
	}
}

func TestCompatibilityDistance(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	a := NewGenome(2, 1, tracker, newTestRand(30))
	b := a.Clone()

	if distance := CompatibilityDistance(a, b, 1, 1, 0.4); distance != 0 {
		t.Fatalf("identical distance = %v, want 0", distance)
	}

	// b drops gene 3 and gains a higher-innovation gene 4.
	b.Connections = b.Connections[:2]
	b.Connections = append(b.Connections, ConnectionGene{Innovation: 99, In: 0, Out: 3, Weight: 1, Enabled: true})

	distance := CompatibilityDistance(a, b, 1, 1, 0.4)
	if distance <= 0 {
		t.Fatalf("distance = %v, want > 0", distance)
	}
}

func TestCrossoverAndMutationStayAcyclic(t *testing.T) {
	tracker := NewInnovationTracker(3, 2)
	options := DefaultOptions(3, 2)
	options.InitialConnectionRate = 0.5
	rng := newTestRand(123)

	population := make([]*Genome, 20)
	for i := range population {
		genome := NewSeededGenome(options, tracker, rng)
		AddNode(genome, tracker, options, rng)
		AddLink(genome, tracker, options, rng)
		population[i] = genome
	}

	for iteration := 0; iteration < 1000; iteration++ {
		mother := population[rng.Intn(len(population))]
		father := population[rng.Intn(len(population))]
		child := Crossover(mother, father, rng)
		Mutate(child, tracker, options, rng)
		if _, err := NewNetwork(child); err != nil {
			t.Fatalf("iteration %d produced an unevaluable child: %v", iteration, err)
		}
	}
}
