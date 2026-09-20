package benchmarks

import (
	"math"

	"github.com/cookiengineer/goneat/engine/backprop"
	"github.com/cookiengineer/goneat/engine/neat"
	"github.com/cookiengineer/goneat/engine/random"
)

// OpenAI Gym CartPole constants.
const (
	cartGravity         = 9.8
	cartMassCart        = 1.0
	cartMassPole        = 0.1
	cartTotalMass       = cartMassCart + cartMassPole
	cartLength          = 0.5
	cartPoleMassLength  = cartMassPole * cartLength
	cartForceMag        = 10.0
	cartTau             = 0.02
	cartXThreshold      = 2.4
	cartThetaThreshold  = 12 * 2 * math.Pi / 360
	cartTrainingSamples = 128
)

// CartPole is the classic single-pole balancing control task. State is
// [x, x_dot, theta, theta_dot]; the network outputs a binary push direction.
// A genome must balance every one of a fixed set of perturbed initial states,
// so a lucky constant policy cannot pass by accident. Fitness is the worst
// number of timesteps balanced across those states.
type CartPole struct {
	maxSteps      int
	initialStates [][4]float64
	samples       []Sample
}

// NewCartPole creates the task with a maximum episode length.
func NewCartPole(maxSteps int) *CartPole {
	if maxSteps <= 0 {
		maxSteps = 500
	}
	return &CartPole{
		maxSteps:      maxSteps,
		initialStates: cartPoleInitialStates(),
		samples:       generateCartPoleSamples(cartTrainingSamples),
	}
}

// cartPoleInitialStates returns a deterministic set of perturbed start states.
func cartPoleInitialStates() [][4]float64 {
	generator := random.New(7)
	states := [][4]float64{{0, 0, 0, 0}}
	for i := 0; i < 3; i++ {
		states = append(states, [4]float64{
			generator.Float64()*1.0 - 0.5,
			generator.Float64()*0.6 - 0.3,
			generator.Float64()*0.12 - 0.06,
			generator.Float64()*0.6 - 0.3,
		})
	}
	return states
}

// Name implements Task.
func (c *CartPole) Name() string { return "cartpole" }

// InputCount implements Task.
func (c *CartPole) InputCount() int { return 4 }

// OutputCount implements Task.
func (c *CartPole) OutputCount() int { return 1 }

// Evaluate implements Task.
func (c *CartPole) Evaluate(genome *neat.Genome) (float64, error) {
	network, err := neat.NewNetwork(genome)
	if err != nil {
		return 0, err
	}
	worst := c.maxSteps
	for _, initial := range c.initialStates {
		steps, err := c.balance(network, initial)
		if err != nil {
			return 0, err
		}
		if steps < worst {
			worst = steps
		}
	}
	return float64(worst), nil
}

func (c *CartPole) balance(network *neat.Network, initial [4]float64) (int, error) {
	outputs := make([]float64, 1)
	state := initial
	for step := 0; step < c.maxSteps; step++ {
		if err := network.Activate(state[:], outputs); err != nil {
			return 0, err
		}
		action := 0
		if outputs[0] >= 0.5 {
			action = 1
		}
		state = stepCartPole(state, action)
		if cartPoleFailed(state) {
			return step + 1, nil
		}
	}
	return c.maxSteps, nil
}

// Solved implements Task.
func (c *CartPole) Solved(genome *neat.Genome) bool {
	fitness, err := c.Evaluate(genome)
	return err == nil && fitness >= float64(c.maxSteps)
}

// Trainer implements Task. There is no ground-truth action in CartPole, so the
// gradient phase imitates a hand-coded balancing heuristic: this is the
// "translate it yourself" signal that gets the search out of a plateau.
func (c *CartPole) Trainer() neat.PhaseTrainer {
	return &SupervisedTrainer{
		Optimizer:   func() backprop.Optimizer { return backprop.NewSGD(0.05, 0.9) },
		WeightRange: 10,
		Samples:     c.samples,
	}
}

// stepCartPole advances the physics one timestep. action 0 pushes left, 1
// pushes right. Equations follow the OpenAI Gym implementation.
func stepCartPole(state [4]float64, action int) [4]float64 {
	x, xDot, theta, thetaDot := state[0], state[1], state[2], state[3]

	force := -cartForceMag
	if action == 1 {
		force = cartForceMag
	}

	cosTheta := math.Cos(theta)
	sinTheta := math.Sin(theta)
	temp := (force + cartPoleMassLength*thetaDot*thetaDot*sinTheta) / cartTotalMass
	thetaAcc := (cartGravity*sinTheta - cosTheta*temp) /
		(cartLength * (4.0/3.0 - cartMassPole*cosTheta*cosTheta/cartTotalMass))
	xAcc := temp - cartPoleMassLength*thetaAcc*cosTheta/cartTotalMass

	x += cartTau * xDot
	xDot += cartTau * xAcc
	theta += cartTau * thetaDot
	thetaDot += cartTau * thetaAcc
	return [4]float64{x, xDot, theta, thetaDot}
}

func cartPoleFailed(state [4]float64) bool {
	return state[0] < -cartXThreshold || state[0] > cartXThreshold ||
		state[2] < -cartThetaThreshold || state[2] > cartThetaThreshold
}

// cartPoleHeuristic is the balancing rule the gradient phase imitates.
func cartPoleHeuristic(state [4]float64) float64 {
	if state[2]+0.5*state[3] > 0 {
		return 1
	}
	return 0
}

func generateCartPoleSamples(count int) []Sample {
	generator := random.New(42)
	samples := make([]Sample, 0, count)
	for i := 0; i < count; i++ {
		state := [4]float64{
			generator.Float64()*2 - 1,
			generator.Float64()*2 - 1,
			generator.Float64()*0.4 - 0.2,
			generator.Float64()*2 - 1,
		}
		samples = append(samples, Sample{
			Inputs:  []float64{state[0], state[1], state[2], state[3]},
			Targets: []float64{cartPoleHeuristic(state)},
		})
	}
	return samples
}
