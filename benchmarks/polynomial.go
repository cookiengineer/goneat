package benchmarks

import (
	"fmt"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
)

// PolynomialClassification is a 2D point classification task. A point (x, y)
// in [-1, 1]^2 is labeled positive when it lies on or above the curve
// y = p(x), where p is a polynomial with an odd leading term (x^3, x^5, ...).
//
// The decision boundary is curved, so it cannot be represented by a single
// linear unit. Solving it requires growing hidden neurons and, in practice,
// exploiting several per-node activation functions (the odd/even powers behave
// differently), which is exactly what this task is meant to demonstrate.
type PolynomialClassification struct {
	name         string
	coefficients []float64 // coefficients[i] * x^i
	points       []Sample
}

// NewPolynomialClassification creates the task for the given boundary
// polynomial and grid resolution. Coefficients are ordered from x^0 upward.
func NewPolynomialClassification(name string, coefficients []float64, resolution int) *PolynomialClassification {
	return NewPolynomialClassificationGrid(name, coefficients, resolution, resolution)
}

// NewPolynomialClassificationGrid creates the task for an explicit grid shape.
// The x and y sample counts are independent because the higher odd powers
// (x^5, x^7) are nearly indistinguishable on a coarse square grid: a slightly
// denser x axis is enough to separate them without paying for a fine square
// grid everywhere.
func NewPolynomialClassificationGrid(name string, coefficients []float64, xResolution, yResolution int) *PolynomialClassification {
	if xResolution < 3 {
		xResolution = 3
	}
	if yResolution < 3 {
		yResolution = 3
	}
	task := &PolynomialClassification{
		name:         name,
		coefficients: append([]float64(nil), coefficients...),
	}

	stepX := 2.0 / float64(xResolution-1)
	stepY := 2.0 / float64(yResolution-1)
	for ix := 0; ix < xResolution; ix++ {
		x := -1 + float64(ix)*stepX
		boundary := task.Boundary(x)
		for iy := 0; iy < yResolution; iy++ {
			y := -1 + float64(iy)*stepY
			label := 0.0
			if y >= boundary {
				label = 1
			}
			task.points = append(task.points, Sample{
				Inputs:  []float64{x, y},
				Targets: []float64{label},
			})
		}
	}
	return task
}

// NewCubicBoundary returns the task with boundary y = x^3.
func NewCubicBoundary(resolution int) *PolynomialClassification {
	return NewPolynomialClassification("cubic", []float64{0, 0, 0, 1}, resolution)
}

// NewQuinticBoundary returns the task with boundary y = x^5.
func NewQuinticBoundary(resolution int) *PolynomialClassification {
	return NewPolynomialClassification("quintic", []float64{0, 0, 0, 0, 0, 1}, resolution)
}

// NewSepticBoundary returns the task with boundary y = x^7.
func NewSepticBoundary(resolution int) *PolynomialClassification {
	return NewPolynomialClassification("septic", []float64{0, 0, 0, 0, 0, 0, 0, 1}, resolution)
}

// Higher-order boundaries are sampled on a denser x axis than y axis. On a
// coarse square grid, x^5 and x^7 produce identical labels because the two
// curves only differ inside a thin band that no sample lands in. The extra x
// resolution exposes the difference without the cost of a fine square grid.
const (
	higherOrderXResolution = 11
	higherOrderYResolution = 9
)

// NewOddPowerBoundary returns the classification task for y = x^power on the
// tuned higher-order grid, where power is an odd exponent >= 3.
func NewOddPowerBoundary(name string, power int) *PolynomialClassification {
	if power < 3 {
		power = 3
	}
	coefficients := make([]float64, power+1)
	coefficients[power] = 1
	return NewPolynomialClassificationGrid(name, coefficients, higherOrderXResolution, higherOrderYResolution)
}

// Boundary evaluates p(x).
func (p *PolynomialClassification) Boundary(x float64) float64 {
	value := 0.0
	power := 1.0
	for _, coefficient := range p.coefficients {
		value += coefficient * power
		power *= x
	}
	return value
}

// Name implements Task.
func (p *PolynomialClassification) Name() string { return p.name }

// InputCount implements Task.
func (p *PolynomialClassification) InputCount() int { return 2 }

// OutputCount implements Task.
func (p *PolynomialClassification) OutputCount() int { return 1 }

// Points returns the labeled dataset.
func (p *PolynomialClassification) Points() []Sample {
	return append([]Sample(nil), p.points...)
}

// Evaluate implements Task.
func (p *PolynomialClassification) Evaluate(genome *neat.Genome) (float64, error) {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return 0, err
	}
	meanSquaredError, _, err := evaluatePatterns(network, p.points)
	if err != nil {
		return 0, err
	}
	return 1 / (1 + meanSquaredError), nil
}

// Solved implements Task.
func (p *PolynomialClassification) Solved(genome *neat.Genome) bool {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return false
	}
	_, accuracy, err := evaluatePatterns(network, p.points)
	return err == nil && accuracy == 1
}

// Accuracy returns the classification accuracy of a genome.
func (p *PolynomialClassification) Accuracy(genome *neat.Genome) float64 {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return 0
	}
	_, accuracy, err := evaluatePatterns(network, p.points)
	if err != nil {
		return 0
	}
	return accuracy
}

// Trainer implements Task.
func (p *PolynomialClassification) Trainer() neat.PhaseTrainer {
	return &SupervisedTrainer{
		Optimizer:   func() backprop.Optimizer { return backprop.NewSGD(0.2, 0.9) },
		WeightRange: 10,
		Samples:     p.points,
	}
}

// Describe returns a human-readable boundary description.
func (p *PolynomialClassification) Describe() string {
	terms := ""
	for power := len(p.coefficients) - 1; power >= 0; power-- {
		coefficient := p.coefficients[power]
		if coefficient == 0 {
			continue
		}
		if terms != "" {
			terms += " + "
		}
		terms += fmt.Sprintf("%g*x^%d", coefficient, power)
	}
	return fmt.Sprintf("y >= %s", terms)
}
