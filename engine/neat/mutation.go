package neat

import "github.com/cookiengineer/goneat/engine/random"

// MutateWeights perturbs or resets connection weights. Each connection is
// handled independently: it is reset with probability WeightResetRate, else
// perturbed with probability WeightPerturbRate.
func MutateWeights(genome *Genome, options Options, rng *random.Rand) {
	for index := range genome.Connections {
		roll := rng.Float64()
		switch {
		case roll < options.WeightResetRate:
			genome.Connections[index].Weight = rng.NormFloat64() * options.InitialWeightStdDev
		case roll < options.WeightResetRate+options.WeightPerturbRate:
			genome.Connections[index].Weight += rng.NormFloat64() * options.WeightPerturbPower
		}
	}
}

// AddLink adds a new connection that does not create a cycle. It returns true
// if a connection was added.
func AddLink(genome *Genome, tracker *InnovationTracker, options Options, rng *random.Rand) bool {
	type candidate struct{ in, out int }
	var candidates []candidate

	for _, source := range genome.Nodes {
		if source.Type == NodeOutput {
			continue
		}
		for _, target := range genome.Nodes {
			if target.Type == NodeInput || target.Type == NodeBias {
				continue
			}
			if source.ID == target.ID || genome.HasConnection(source.ID, target.ID) {
				continue
			}
			if reaches(genome, target.ID, source.ID) {
				continue
			}
			candidates = append(candidates, candidate{in: source.ID, out: target.ID})
		}
	}

	if len(candidates) == 0 {
		return false
	}
	choice := candidates[rng.Intn(len(candidates))]
	genome.Connections = append(genome.Connections, ConnectionGene{
		Innovation: tracker.ConnectionInnovation(choice.in, choice.out),
		In:         choice.in,
		Out:        choice.out,
		Weight:     rng.NormFloat64() * options.InitialWeightStdDev,
		Enabled:    true,
	})
	return true
}

// AddNode splits a random enabled connection into two, inserting a new hidden
// node with a random activation. It returns true if a node was added.
func AddNode(genome *Genome, tracker *InnovationTracker, options Options, rng *random.Rand) bool {
	var enabled []int
	for index := range genome.Connections {
		if genome.Connections[index].Enabled {
			enabled = append(enabled, index)
		}
	}
	if len(enabled) == 0 {
		return false
	}

	connectionIndex := enabled[rng.Intn(len(enabled))]
	connection := genome.Connections[connectionIndex]
	genome.Connections[connectionIndex].Enabled = false

	newNodeID := tracker.NextNodeID()
	activation := ActivationSigmoid
	if len(options.ActivationOptions) > 0 {
		activation = options.ActivationOptions[rng.Intn(len(options.ActivationOptions))]
	}
	genome.Nodes = append(genome.Nodes, NodeGene{
		ID:         newNodeID,
		Type:       NodeHidden,
		Activation: activation,
	})

	genome.Connections = append(genome.Connections,
		ConnectionGene{
			Innovation: tracker.ConnectionInnovation(connection.In, newNodeID),
			In:         connection.In,
			Out:        newNodeID,
			Weight:     1.0,
			Enabled:    true,
		},
		ConnectionGene{
			Innovation: tracker.ConnectionInnovation(newNodeID, connection.Out),
			In:         newNodeID,
			Out:        connection.Out,
			Weight:     connection.Weight,
			Enabled:    true,
		},
	)
	return true
}

// ToggleEnable flips the enabled state of a random connection, returning true
// if a connection changed.
func ToggleEnable(genome *Genome, rng *random.Rand) bool {
	if len(genome.Connections) == 0 {
		return false
	}
	index := rng.Intn(len(genome.Connections))
	genome.Connections[index].Enabled = !genome.Connections[index].Enabled
	return true
}

// Mutate applies weight mutation plus the configured structural mutations
// using the default (complexifying) phase.
func Mutate(genome *Genome, tracker *InnovationTracker, options Options, rng *random.Rand) {
	MutateWithPhase(genome, tracker, options, rng, PhaseDefault)
}

// MutateWithPhase applies mutations appropriate to the active search phase.
//
//   - PhaseDefault and PhaseHybrid complexify (add links/nodes/toggle).
//   - PhasePrune simplifies (remove links/nodes).
//   - PhaseBackprop only perturbs weights; training happens separately.
func MutateWithPhase(genome *Genome, tracker *InnovationTracker, options Options, rng *random.Rand, phase Phase) {
	MutateWeights(genome, options, rng)

	switch phase {
	case PhasePrune:
		if rng.Float64() < options.PruneLinkRate {
			PruneLink(genome, rng)
		}
		if rng.Float64() < options.PruneNodeRate {
			PruneNode(genome, rng)
		}
	case PhaseBackprop:
		// Weight-only exploitation; gradient training is applied by the
		// population after reproduction.
	default:
		if rng.Float64() < options.AddLinkRate {
			AddLink(genome, tracker, options, rng)
		}
		if rng.Float64() < options.AddNodeRate {
			AddNode(genome, tracker, options, rng)
		}
		if rng.Float64() < options.ToggleEnableRate {
			ToggleEnable(genome, rng)
		}
	}
}

// PruneLink removes a random enabled connection, returning true if one was
// removed. Removing genes shrinks the genome, which is the simplifying half of
// phased searching.
func PruneLink(genome *Genome, rng *random.Rand) bool {
	enabled := make([]int, 0, len(genome.Connections))
	for index := range genome.Connections {
		if genome.Connections[index].Enabled {
			enabled = append(enabled, index)
		}
	}
	if len(enabled) == 0 {
		return false
	}
	index := enabled[rng.Intn(len(enabled))]
	genome.Connections = append(genome.Connections[:index], genome.Connections[index+1:]...)
	return true
}

// PruneNode removes a random hidden node together with every connection that
// touches it, returning true if one was removed. Input, bias and output nodes
// are never removed.
func PruneNode(genome *Genome, rng *random.Rand) bool {
	hidden := genome.HiddenIDs()
	if len(hidden) == 0 {
		return false
	}
	nodeID := hidden[rng.Intn(len(hidden))]

	nodes := genome.Nodes[:0]
	for _, node := range genome.Nodes {
		if node.ID != nodeID {
			nodes = append(nodes, node)
		}
	}
	genome.Nodes = nodes

	connections := genome.Connections[:0]
	for _, connection := range genome.Connections {
		if connection.In != nodeID && connection.Out != nodeID {
			connections = append(connections, connection)
		}
	}
	genome.Connections = connections
	return true
}

// reaches reports whether target is reachable from source by following enabled
// connections.
func reaches(genome *Genome, source, target int) bool {
	if source == target {
		return true
	}
	adjacency := make(map[int][]int)
	for _, connection := range genome.Connections {
		if connection.Enabled {
			adjacency[connection.In] = append(adjacency[connection.In], connection.Out)
		}
	}

	visited := map[int]bool{source: true}
	queue := []int{source}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, next := range adjacency[current] {
			if next == target {
				return true
			}
			if !visited[next] {
				visited[next] = true
				queue = append(queue, next)
			}
		}
	}
	return false
}
