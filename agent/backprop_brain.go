package agent

import (
	"fmt"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/engine/random"
)

// BackpropBrain is a fixed-topology feed-forward network trained purely by
// gradient descent. It does not participate in NEAT mutation; it exists for
// tasks where the topology is known and only the weights need to be learned.
//
// Internally it reuses NEATBrain over a constructed genome, so gradient
// learning, cloning and snapshots behave identically.
type BackpropBrain struct {
	*NEATBrain
}

// NewBackpropBrain builds a single-hidden-layer network with the given
// dimensions. A hidden count of zero produces a direct input-to-output map.
func NewBackpropBrain(
	inputDimensions, outputDimensions, hiddenDimensions int,
	hiddenActivation neat.ActivationType,
	optimizer backprop.Optimizer,
	weightRange float64,
	rng *random.Rand,
) (*BackpropBrain, error) {
	if inputDimensions <= 0 || outputDimensions <= 0 {
		return nil, fmt.Errorf("agent: invalid backprop dimensions (%d -> %d)", inputDimensions, outputDimensions)
	}
	if hiddenDimensions < 0 {
		return nil, fmt.Errorf("agent: hidden dimensions must be non-negative")
	}
	if optimizer == nil {
		optimizer = backprop.NewSGD(0.1, 0.9)
	}
	if rng == nil {
		rng = random.New(1)
	}

	tracker := neat.NewInnovationTracker(inputDimensions, outputDimensions)
	genome := &neat.Genome{ID: -1}

	for input := 0; input < inputDimensions; input++ {
		genome.Nodes = append(genome.Nodes, neat.NodeGene{ID: input, Type: neat.NodeInput, Activation: neat.ActivationLinear})
	}
	biasID := inputDimensions
	genome.Nodes = append(genome.Nodes, neat.NodeGene{ID: biasID, Type: neat.NodeBias, Activation: neat.ActivationLinear})

	hiddenIDs := make([]int, 0, hiddenDimensions)
	for hidden := 0; hidden < hiddenDimensions; hidden++ {
		id := tracker.NextNodeID()
		hiddenIDs = append(hiddenIDs, id)
		genome.Nodes = append(genome.Nodes, neat.NodeGene{ID: id, Type: neat.NodeHidden, Activation: hiddenActivation})
	}

	outputIDs := make([]int, 0, outputDimensions)
	for output := 0; output < outputDimensions; output++ {
		id := inputDimensions + 1 + output
		outputIDs = append(outputIDs, id)
		genome.Nodes = append(genome.Nodes, neat.NodeGene{ID: id, Type: neat.NodeOutput, Activation: neat.ActivationSigmoid})
	}

	addConnection := func(in, out int) {
		genome.Connections = append(genome.Connections, neat.ConnectionGene{
			Innovation: tracker.ConnectionInnovation(in, out),
			In:         in,
			Out:        out,
			Weight:     rng.NormFloat64(),
			Enabled:    true,
		})
	}

	if len(hiddenIDs) == 0 {
		for source := 0; source <= inputDimensions; source++ {
			for _, out := range outputIDs {
				addConnection(source, out)
			}
		}
	} else {
		for source := 0; source <= inputDimensions; source++ {
			for _, hidden := range hiddenIDs {
				addConnection(source, hidden)
			}
		}
		for _, hidden := range hiddenIDs {
			for _, out := range outputIDs {
				addConnection(hidden, out)
			}
		}
		for _, out := range outputIDs {
			addConnection(biasID, out)
		}
	}

	brain, err := NewNEATBrainWithTrainer(genome, backprop.NewTrainer(optimizer, weightRange), backprop.MSELoss{})
	if err != nil {
		return nil, err
	}
	return &BackpropBrain{NEATBrain: brain}, nil
}
