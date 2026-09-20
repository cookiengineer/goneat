package neat

import (
	"sort"

	"github.com/cookiengineer/goneat/engine/random"
)

// Genome is a genotype: a set of nodes and weighted connections. The same
// struct is the single source of truth for genetic operators, backpropagation
// and JSON snapshots.
type Genome struct {
	ID          int              `json:"id"`
	Nodes       []NodeGene       `json:"nodes"`
	Connections []ConnectionGene `json:"connections"`
	Fitness     float64          `json:"fitness"`
	SpeciesID   int              `json:"species_id"`
	Generation  int              `json:"generation"`
	Parents     []int            `json:"parents,omitempty"`
}

// NewGenome builds a fully connected feed-forward genome with one bias node
// using default options.
func NewGenome(inputCount, outputCount int, tracker *InnovationTracker, rng *random.Rand) *Genome {
	options := DefaultOptions(inputCount, outputCount)
	options.InitialConnectionRate = 1.0
	return NewSeededGenome(options, tracker, rng)
}

// NewSeededGenome builds a feed-forward genome honoring the initial
// connection rate, weight spread and output activation from options.
func NewSeededGenome(options Options, tracker *InnovationTracker, rng *random.Rand) *Genome {
	genome := &Genome{ID: -1}

	for input := 0; input < options.InputCount; input++ {
		genome.Nodes = append(genome.Nodes, NodeGene{ID: input, Type: NodeInput, Activation: ActivationLinear})
	}
	biasID := options.InputCount
	genome.Nodes = append(genome.Nodes, NodeGene{ID: biasID, Type: NodeBias, Activation: ActivationLinear})
	for output := 0; output < options.OutputCount; output++ {
		genome.Nodes = append(genome.Nodes, NodeGene{
			ID:         options.InputCount + 1 + output,
			Type:       NodeOutput,
			Activation: options.OutputActivation,
		})
	}

	for output := 0; output < options.OutputCount; output++ {
		outID := options.InputCount + 1 + output
		for source := 0; source <= options.InputCount; source++ {
			if options.InitialConnectionRate < 1 && rng.Float64() >= options.InitialConnectionRate {
				continue
			}
			genome.Connections = append(genome.Connections, ConnectionGene{
				Innovation: tracker.ConnectionInnovation(source, outID),
				In:         source,
				Out:        outID,
				Weight:     rng.NormFloat64() * options.InitialWeightStdDev,
				Enabled:    true,
			})
		}
	}

	return genome
}

// Clone returns a deep copy of the genome.
func (g *Genome) Clone() *Genome {
	if g == nil {
		return nil
	}
	clone := &Genome{
		ID:          g.ID,
		Fitness:     g.Fitness,
		SpeciesID:   g.SpeciesID,
		Generation:  g.Generation,
		Nodes:       append([]NodeGene(nil), g.Nodes...),
		Connections: append([]ConnectionGene(nil), g.Connections...),
		Parents:     append([]int(nil), g.Parents...),
	}
	return clone
}

// NodeIndex returns the index of a node in the Nodes slice, or -1.
func (g *Genome) NodeIndex(id int) int {
	for i := range g.Nodes {
		if g.Nodes[i].ID == id {
			return i
		}
	}
	return -1
}

// NodeByID returns a pointer to a node, or nil.
func (g *Genome) NodeByID(id int) *NodeGene {
	if index := g.NodeIndex(id); index >= 0 {
		return &g.Nodes[index]
	}
	return nil
}

// HasNode reports whether a node ID exists.
func (g *Genome) HasNode(id int) bool {
	return g.NodeIndex(id) >= 0
}

// InputIDs returns the sorted input node IDs.
func (g *Genome) InputIDs() []int {
	return g.idsByType(NodeInput)
}

// OutputIDs returns the sorted output node IDs.
func (g *Genome) OutputIDs() []int {
	return g.idsByType(NodeOutput)
}

// HiddenIDs returns the sorted hidden node IDs.
func (g *Genome) HiddenIDs() []int {
	return g.idsByType(NodeHidden)
}

// BiasID returns the bias node ID and whether one exists.
func (g *Genome) BiasID() (int, bool) {
	for i := range g.Nodes {
		if g.Nodes[i].Type == NodeBias {
			return g.Nodes[i].ID, true
		}
	}
	return 0, false
}

func (g *Genome) idsByType(nodeType NodeType) []int {
	var ids []int
	for i := range g.Nodes {
		if g.Nodes[i].Type == nodeType {
			ids = append(ids, g.Nodes[i].ID)
		}
	}
	sort.Ints(ids)
	return ids
}

// ConnectionIndex returns the index of a connection by innovation, or -1.
func (g *Genome) ConnectionIndex(innovation int) int {
	for i := range g.Connections {
		if g.Connections[i].Innovation == innovation {
			return i
		}
	}
	return -1
}

// HasConnection reports whether a connection from in to out exists.
func (g *Genome) HasConnection(in, out int) bool {
	for i := range g.Connections {
		if g.Connections[i].In == in && g.Connections[i].Out == out {
			return true
		}
	}
	return false
}

// EnabledConnectionCount counts enabled connections.
func (g *Genome) EnabledConnectionCount() int {
	count := 0
	for i := range g.Connections {
		if g.Connections[i].Enabled {
			count++
		}
	}
	return count
}

// WeightVector returns enabled connection weights in innovation order.
func (g *Genome) WeightVector() []float64 {
	ordered := append([]ConnectionGene(nil), g.Connections...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Innovation < ordered[j].Innovation })
	weights := make([]float64, 0, len(ordered))
	for i := range ordered {
		if ordered[i].Enabled {
			weights = append(weights, ordered[i].Weight)
		}
	}
	return weights
}

// HiddenActivationCounts returns a histogram of activation functions used by
// hidden nodes.
func (g *Genome) HiddenActivationCounts() map[ActivationType]int {
	counts := make(map[ActivationType]int)
	for _, node := range g.Nodes {
		if node.Type == NodeHidden {
			counts[node.Activation]++
		}
	}
	return counts
}

// DistinctHiddenActivations counts how many different activation functions the
// hidden nodes use.
func (g *Genome) DistinctHiddenActivations() int {
	return len(g.HiddenActivationCounts())
}
