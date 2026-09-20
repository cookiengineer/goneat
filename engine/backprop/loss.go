// Package backprop provides gradient training over an evolved neat.Genome.
//
// Standard NEAT evolves topology and weights together. goneat additionally
// allows gradient descent to refine the weights of an evolved topology, which
// is the hybrid learning model used by the simulation. The graph is built from
// a neat.Network, gradients are accumulated per connection innovation, and the
// updated weights are written straight back into the genome so genetic and
// gradient learning share one source of truth.
package backprop

import (
	"errors"
	"math"
)

// Loss computes a scalar loss and its derivative with respect to the network
// outputs. The gradient slice is overwritten in place.
type Loss interface {
	Compute(outputs, targets []float64, gradient []float64) (loss float64, err error)
}

// MSELoss is the mean squared error.
type MSELoss struct{}

// Compute implements Loss.
func (MSELoss) Compute(outputs, targets []float64, gradient []float64) (float64, error) {
	if len(outputs) != len(targets) {
		return 0, errors.New("backprop: outputs and targets differ in length")
	}
	if len(outputs) == 0 {
		return 0, errors.New("backprop: empty output")
	}
	scale := 1 / float64(len(outputs))
	loss := 0.0
	for i := range outputs {
		difference := outputs[i] - targets[i]
		loss += difference * difference
		if gradient != nil {
			gradient[i] = 2 * difference * scale
		}
	}
	return loss * scale, nil
}

// CrossEntropyLoss treats outputs as probabilities in (0, 1), typically from a
// sigmoid output activation.
type CrossEntropyLoss struct{}

// Compute implements Loss.
func (CrossEntropyLoss) Compute(outputs, targets []float64, gradient []float64) (float64, error) {
	if len(outputs) != len(targets) {
		return 0, errors.New("backprop: outputs and targets differ in length")
	}
	if len(outputs) == 0 {
		return 0, errors.New("backprop: empty output")
	}
	const epsilon = 1e-12
	scale := 1 / float64(len(outputs))
	loss := 0.0
	for i := range outputs {
		output := math.Min(1-epsilon, math.Max(epsilon, outputs[i]))
		loss += -(targets[i]*math.Log(output) + (1-targets[i])*math.Log(1-output))
		if gradient != nil {
			gradient[i] = (output - targets[i]) / (output * (1 - output)) * scale
		}
	}
	return loss * scale, nil
}
