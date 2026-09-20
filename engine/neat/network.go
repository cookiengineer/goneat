package neat

import "errors"

// ErrCyclicNetwork indicates the genome contains a cycle. Recurrent topologies
// are not evaluated by the feed-forward Network; a later phase adds bounded
// relaxation for them.
var ErrCyclicNetwork = errors.New("neat: genome contains a cycle")

// Network is the phenotype of a genome: a topologically ordered evaluation
// plan. Build one per genome and reuse it across activations; it owns its
// value buffer and performs no allocations during Activate.
type Network struct {
	genome    *Genome
	order     []int
	incoming  map[int][]int
	inputIDs  []int
	outputIDs []int
	biasID    int
	hasBias   bool
	values    map[int]float64
}

// NewNetwork builds and topologically sorts the evaluation plan.
func NewNetwork(genome *Genome) (*Network, error) {
	network := &Network{
		genome:   genome,
		incoming: make(map[int][]int),
		values:   make(map[int]float64, len(genome.Nodes)),
	}
	network.inputIDs = genome.InputIDs()
	network.outputIDs = genome.OutputIDs()
	network.biasID, network.hasBias = genome.BiasID()

	for index, connection := range genome.Connections {
		if connection.Enabled {
			network.incoming[connection.Out] = append(network.incoming[connection.Out], index)
		}
	}

	order, err := topologicalOrder(genome)
	if err != nil {
		return nil, err
	}
	network.order = order
	return network, nil
}

// topologicalOrder returns node IDs in a valid evaluation order using Kahn's
// algorithm over enabled connections.
func topologicalOrder(genome *Genome) ([]int, error) {
	indegree := make(map[int]int, len(genome.Nodes))
	adjacency := make(map[int][]int, len(genome.Nodes))
	for _, node := range genome.Nodes {
		indegree[node.ID] = 0
	}
	for _, connection := range genome.Connections {
		if !connection.Enabled {
			continue
		}
		adjacency[connection.In] = append(adjacency[connection.In], connection.Out)
		indegree[connection.Out]++
	}

	queue := make([]int, 0, len(genome.Nodes))
	for _, node := range genome.Nodes {
		if indegree[node.ID] == 0 {
			queue = append(queue, node.ID)
		}
	}

	order := make([]int, 0, len(genome.Nodes))
	for len(queue) > 0 {
		// Keep the frontier sorted so evaluation order is deterministic across
		// runs and map iteration order.
		sortInts(queue)
		id := queue[0]
		queue = queue[1:]
		order = append(order, id)

		next := adjacency[id]
		sortInts(next)
		for _, target := range next {
			indegree[target]--
			if indegree[target] == 0 {
				queue = append(queue, target)
			}
		}
	}

	if len(order) != len(genome.Nodes) {
		return nil, ErrCyclicNetwork
	}
	return order, nil
}

func sortInts(values []int) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}

// Genome returns the underlying genotype.
func (n *Network) Genome() *Genome { return n.genome }

// InputCount returns the number of input nodes.
func (n *Network) InputCount() int { return len(n.inputIDs) }

// OutputCount returns the number of output nodes.
func (n *Network) OutputCount() int { return len(n.outputIDs) }

// InputIDs returns the input node IDs in frame order.
func (n *Network) InputIDs() []int { return n.inputIDs }

// OutputIDs returns the output node IDs in frame order.
func (n *Network) OutputIDs() []int { return n.outputIDs }

// Order returns the evaluation order as node IDs.
func (n *Network) Order() []int { return n.order }

// IncomingIndices returns the indices of enabled connections targeting a node.
// Use Connection to read the gene; weights therefore always reflect the latest
// genome state without rebuilding the network.
func (n *Network) IncomingIndices(id int) []int { return n.incoming[id] }

// Connection returns the connection gene at the given genome index.
func (n *Network) Connection(index int) ConnectionGene {
	return n.genome.Connections[index]
}

// Incoming returns copies of the enabled connections targeting a node.
func (n *Network) Incoming(id int) []ConnectionGene {
	indices := n.incoming[id]
	if len(indices) == 0 {
		return nil
	}
	connections := make([]ConnectionGene, 0, len(indices))
	for _, index := range indices {
		connections = append(connections, n.genome.Connections[index])
	}
	return connections
}

// Node returns a node gene by ID.
func (n *Network) Node(id int) NodeGene {
	if node := n.genome.NodeByID(id); node != nil {
		return *node
	}
	return NodeGene{ID: id, Type: NodeHidden, Activation: ActivationLinear}
}

// Value returns the last computed activation of a node.
func (n *Network) Value(id int) float64 { return n.values[id] }

// Activate evaluates the network for the given inputs and writes the outputs.
func (n *Network) Activate(inputs []float64, outputs []float64) error {
	if len(inputs) < len(n.inputIDs) {
		return errors.New("neat: not enough inputs for network")
	}
	if len(outputs) < len(n.outputIDs) {
		return errors.New("neat: output buffer too small")
	}

	for id := range n.values {
		n.values[id] = 0
	}
	for index, id := range n.inputIDs {
		n.values[id] = inputs[index]
	}
	if n.hasBias {
		n.values[n.biasID] = 1
	}

	for _, id := range n.order {
		node := n.genome.NodeByID(id)
		if node == nil || node.Type == NodeInput || node.Type == NodeBias {
			continue
		}
		n.values[id] = n.evaluateNode(*node)
	}

	for index, id := range n.outputIDs {
		outputs[index] = n.values[id]
	}
	return nil
}

func (n *Network) evaluateNode(node NodeGene) float64 {
	indices := n.incoming[node.ID]
	if len(indices) == 0 {
		return node.Activation.Apply(0)
	}

	if node.Activation == ActivationMult {
		product := 1.0
		for _, index := range indices {
			connection := n.genome.Connections[index]
			product *= connection.Weight * n.values[connection.In]
		}
		return product
	}

	sum := 0.0
	for _, index := range indices {
		connection := n.genome.Connections[index]
		sum += connection.Weight * n.values[connection.In]
	}
	return node.Activation.Apply(sum)
}

// BreakCycles removes enabled connections until the genome is acyclic. Crossover
// can combine two acyclic parents into a child that contains a cycle; this
// keeps every genome evaluable as a feed-forward network (and differentiable by
// the backprop graph). It returns the number of connections removed.
func BreakCycles(genome *Genome) int {
	removed := 0
	for {
		order, err := topologicalOrder(genome)
		if err == nil {
			return removed
		}
		inOrder := make(map[int]bool, len(order))
		for _, id := range order {
			inOrder[id] = true
		}

		// Drop the newest enabled connection whose target is stuck in a cycle.
		index := -1
		for i := len(genome.Connections) - 1; i >= 0; i-- {
			connection := genome.Connections[i]
			if connection.Enabled && !inOrder[connection.Out] {
				index = i
				break
			}
		}
		if index == -1 {
			return removed
		}
		genome.Connections = append(genome.Connections[:index], genome.Connections[index+1:]...)
		removed++
	}
}
