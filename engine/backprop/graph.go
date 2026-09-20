package backprop

import (
	"errors"

	"github.com/cookiengineer/goneat/engine/neat"
)

// Graph is a differentiable evaluation of a neat.Network. Build one per
// topology and reuse it across Forward/Backward calls; buffers are retained.
type Graph struct {
	network  *neat.Network
	order    []int
	values   map[int]float64
	pre      map[int]float64
	dValues  map[int]float64
	dWeights map[int]float64
	outputs  []float64
	gradient []float64
}

// NewGraph creates a graph for the given network.
func NewGraph(network *neat.Network) *Graph {
	return &Graph{
		network:  network,
		order:    network.Order(),
		values:   make(map[int]float64, len(network.Order())),
		pre:      make(map[int]float64, len(network.Order())),
		dValues:  make(map[int]float64, len(network.Order())),
		dWeights: make(map[int]float64, len(network.Genome().Connections)),
		outputs:  make([]float64, network.OutputCount()),
		gradient: make([]float64, network.OutputCount()),
	}
}

// Network returns the underlying network.
func (g *Graph) Network() *neat.Network { return g.network }

// Forward evaluates the network and returns the output buffer. The returned
// slice is owned by the graph and is valid until the next Forward call.
func (g *Graph) Forward(inputs []float64) ([]float64, error) {
	if len(inputs) < g.network.InputCount() {
		return nil, errors.New("backprop: not enough inputs for network")
	}

	for key := range g.values {
		g.values[key] = 0
	}
	for key := range g.pre {
		g.pre[key] = 0
	}

	for index, id := range g.network.InputIDs() {
		g.values[id] = inputs[index]
	}
	if bias, ok := g.network.Genome().BiasID(); ok {
		g.values[bias] = 1
	}

	for _, id := range g.order {
		node := g.network.Node(id)
		if node.Type == neat.NodeInput || node.Type == neat.NodeBias {
			continue
		}
		g.evaluate(node)
	}

	for index, id := range g.network.OutputIDs() {
		g.outputs[index] = g.values[id]
	}
	return g.outputs, nil
}

func (g *Graph) evaluate(node neat.NodeGene) {
	indices := g.network.IncomingIndices(node.ID)
	if len(indices) == 0 {
		g.values[node.ID] = node.Activation.Apply(0)
		g.pre[node.ID] = 0
		return
	}

	if node.Activation == neat.ActivationMult {
		product := 1.0
		for _, index := range indices {
			connection := g.network.Connection(index)
			product *= connection.Weight * g.values[connection.In]
		}
		g.values[node.ID] = product
		g.pre[node.ID] = product
		return
	}

	sum := 0.0
	for _, index := range indices {
		connection := g.network.Connection(index)
		sum += connection.Weight * g.values[connection.In]
	}
	g.pre[node.ID] = sum
	g.values[node.ID] = node.Activation.Apply(sum)
}

// Backward computes gradients for the given targets and returns the loss. It
// must be called after Forward.
func (g *Graph) Backward(loss Loss, targets []float64) (float64, error) {
	if len(targets) != len(g.outputs) {
		return 0, errors.New("backprop: targets do not match output count")
	}
	value, err := loss.Compute(g.outputs, targets, g.gradient)
	if err != nil {
		return 0, err
	}
	g.ZeroGradients()
	g.AccumulateOutputGradient(g.gradient)
	return value, nil
}

// ZeroGradients clears the accumulated dLoss/dValue and dLoss/dWeight state.
func (g *Graph) ZeroGradients() {
	for key := range g.dValues {
		g.dValues[key] = 0
	}
	for key := range g.dWeights {
		g.dWeights[key] = 0
	}
}

// AccumulateOutputGradient seeds the output layer with the supplied gradient
// and propagates it into the weight gradients. Unlike Backward it does not
// clear dWeights, so repeated calls accumulate parameter gradients across
// several Forward passes. This is what enables differentiating a function that
// queries a network many times (for example, a CPPN evaluated per substrate
// connection).
func (g *Graph) AccumulateOutputGradient(outputGradient []float64) {
	for key := range g.dValues {
		g.dValues[key] = 0
	}
	for index, id := range g.network.OutputIDs() {
		if index < len(outputGradient) {
			g.dValues[id] += outputGradient[index]
		}
	}
	for position := len(g.order) - 1; position >= 0; position-- {
		id := g.order[position]
		node := g.network.Node(id)
		if node.Type == neat.NodeInput || node.Type == neat.NodeBias {
			continue
		}
		g.accumulate(node)
	}
}

func (g *Graph) accumulate(node neat.NodeGene) {
	downstream := g.dValues[node.ID]
	if downstream == 0 {
		return
	}
	indices := g.network.IncomingIndices(node.ID)
	if len(indices) == 0 {
		return
	}

	if node.Activation == neat.ActivationMult {
		// value = prod(weight_i * input_i); differentiate the product.
		terms := make([]float64, len(indices))
		for i, index := range indices {
			connection := g.network.Connection(index)
			terms[i] = connection.Weight * g.values[connection.In]
		}
		for i, index := range indices {
			connection := g.network.Connection(index)
			others := 1.0
			for j := range terms {
				if j != i {
					others *= terms[j]
				}
			}
			g.dValues[connection.In] += downstream * others * connection.Weight
			g.dWeights[connection.Innovation] += downstream * others * g.values[connection.In]
		}
		return
	}

	derivative := downstream * node.Activation.Derivative(g.pre[node.ID])
	for _, index := range indices {
		connection := g.network.Connection(index)
		g.dValues[connection.In] += derivative * connection.Weight
		g.dWeights[connection.Innovation] += derivative * g.values[connection.In]
	}
}

// WeightGradients returns dLoss/dWeight keyed by connection innovation.
func (g *Graph) WeightGradients() map[int]float64 { return g.dWeights }

// Values returns the last computed node activations keyed by node ID.
func (g *Graph) Values() map[int]float64 { return g.values }
