package neat

import "testing"

func TestPruneLinkRemovesConnection(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(1))
	before := len(genome.Connections)

	if !PruneLink(genome, newTestRand(2)) {
		t.Fatalf("PruneLink returned false")
	}
	if len(genome.Connections) != before-1 {
		t.Fatalf("connections = %d, want %d", len(genome.Connections), before-1)
	}
	if _, err := NewNetwork(genome); err != nil {
		t.Fatalf("network after prune: %v", err)
	}
}

func TestPruneNodeRemovesHiddenAndConnections(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(3))
	for i := 0; i < 20 && len(genome.HiddenIDs()) == 0; i++ {
		AddNode(genome, tracker, DefaultOptions(2, 1), newTestRand(int64(100+i)))
	}
	if len(genome.HiddenIDs()) == 0 {
		t.Fatalf("failed to add a hidden node")
	}
	hidden := genome.HiddenIDs()[0]

	if !PruneNode(genome, newTestRand(4)) {
		t.Fatalf("PruneNode returned false")
	}
	if genome.HasNode(hidden) {
		t.Fatalf("hidden node %d still present", hidden)
	}
	for _, connection := range genome.Connections {
		if connection.In == hidden || connection.Out == hidden {
			t.Fatalf("connection touching pruned node remains")
		}
	}
	if _, err := NewNetwork(genome); err != nil {
		t.Fatalf("network after node prune: %v", err)
	}
}

func TestPruneNodeNoHiddenIsNoOp(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(5))
	if PruneNode(genome, newTestRand(6)) {
		t.Fatalf("PruneNode should be a no-op without hidden nodes")
	}
}

func TestBackpropPhaseOnlyChangesWeights(t *testing.T) {
	tracker := NewInnovationTracker(2, 1)
	genome := NewGenome(2, 1, tracker, newTestRand(7))
	nodesBefore := len(genome.Nodes)
	connectionsBefore := len(genome.Connections)
	weightsBefore := genome.WeightVector()

	options := DefaultOptions(2, 1)
	options.WeightPerturbRate = 1.0
	options.WeightResetRate = 0
	MutateWithPhase(genome, tracker, options, newTestRand(8), PhaseBackprop)

	if len(genome.Nodes) != nodesBefore || len(genome.Connections) != connectionsBefore {
		t.Fatalf("backprop phase changed topology: %d/%d nodes, %d/%d connections",
			len(genome.Nodes), nodesBefore, len(genome.Connections), connectionsBefore)
	}
	weightsAfter := genome.WeightVector()
	changed := false
	for i := range weightsBefore {
		if weightsBefore[i] != weightsAfter[i] {
			changed = true
		}
	}
	if !changed {
		t.Fatalf("backprop phase did not perturb weights")
	}
}

func TestPrunePhaseShrinks(t *testing.T) {
	tracker := NewInnovationTracker(4, 2)
	options := DefaultOptions(4, 2)
	options.InitialConnectionRate = 0.5
	genome := NewSeededGenome(options, tracker, newTestRand(9))

	before := len(genome.Connections)
	options.PruneLinkRate = 1.0
	options.PruneNodeRate = 0
	for i := 0; i < 50; i++ {
		MutateWithPhase(genome, tracker, options, newTestRand(int64(200+i)), PhasePrune)
	}
	if len(genome.Connections) >= before {
		t.Fatalf("prune phase did not shrink the genome: %d -> %d", before, len(genome.Connections))
	}
}

func TestPhaseControllerAlternates(t *testing.T) {
	controller := newPhaseController(PhaseOptions{Enabled: true, Duration: 2, StallLimit: 0})
	if controller.current() != PhaseDefault {
		t.Fatalf("initial phase = %v", controller.current())
	}
	controller.observe(1.0)
	if controller.current() != PhaseDefault {
		t.Fatalf("advanced too early: %v", controller.current())
	}
	controller.observe(1.0) // elapsed reaches 2 -> advance
	if controller.current() != PhasePrune {
		t.Fatalf("expected prune phase, got %v", controller.current())
	}
	controller.observe(1.0)
	controller.observe(1.0)
	if controller.current() != PhaseDefault {
		t.Fatalf("expected wrap to default, got %v", controller.current())
	}
}

func TestPhaseControllerStallAdvances(t *testing.T) {
	controller := newPhaseController(PhaseOptions{Enabled: true, Duration: 0, StallLimit: 3})
	// The first observation establishes the baseline; three further
	// non-improving generations trigger the switch.
	for i := 0; i < 4; i++ {
		controller.observe(1.0)
	}
	if controller.current() != PhasePrune {
		t.Fatalf("stall did not advance phase: %v", controller.current())
	}
}

func TestPopulationTrainsDuringBackpropPhase(t *testing.T) {
	options := DefaultOptions(2, 1)
	options.PopulationSize = 6
	options.Seed = 11
	options.Phases = PhaseOptions{
		Enabled:         true,
		Schedule:        []Phase{PhaseBackprop},
		TrainIterations: 3,
	}

	calls := 0
	options.Phases.Trainer = PhaseTrainerFunc(func(genome *Genome, iterations int) error {
		calls++
		for index := range genome.Connections {
			genome.Connections[index].Weight += 0.001
		}
		return nil
	})

	population, err := NewPopulation(options)
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}
	if _, err := population.Epoch(EvaluatorFunc(xorEvaluator), nil); err != nil {
		t.Fatalf("Epoch: %v", err)
	}
	if calls != options.PopulationSize {
		t.Fatalf("trainer called %d times, want %d", calls, options.PopulationSize)
	}
	if population.Phase() != PhaseBackprop {
		t.Fatalf("phase = %v, want backprop", population.Phase())
	}
}

func TestPhaseOptionsSurviveSnapshot(t *testing.T) {
	options := DefaultOptions(2, 1)
	options.PopulationSize = 6
	options.Seed = 3
	options.Phases = PhaseOptions{
		Enabled:    true,
		Duration:   3,
		StallLimit: 2,
		Schedule:   []Phase{PhaseDefault, PhasePrune},
	}
	population, err := NewPopulation(options)
	if err != nil {
		t.Fatalf("NewPopulation: %v", err)
	}
	if _, err := population.Epoch(EvaluatorFunc(xorEvaluator), nil); err != nil {
		t.Fatalf("Epoch: %v", err)
	}

	data, err := population.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	restored, err := DecodePopulation(data)
	if err != nil {
		t.Fatalf("DecodePopulation: %v", err)
	}
	if restored.Phase() != population.Phase() {
		t.Fatalf("phase not preserved: %v vs %v", restored.Phase(), population.Phase())
	}
	if _, err := restored.Epoch(EvaluatorFunc(xorEvaluator), nil); err != nil {
		t.Fatalf("restored Epoch: %v", err)
	}
}
