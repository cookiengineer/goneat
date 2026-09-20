package backprop

import (
	"encoding/json"
	"math"
	"strconv"
)

// Optimizer updates a single parameter given its gradient. State is keyed by
// connection innovation so it survives topology changes and snapshots.
type Optimizer interface {
	// Kind identifies the optimizer implementation for serialization.
	Kind() string
	// Update returns the new value of the parameter.
	Update(key int, parameter, gradient float64) float64
}

// SGD is stochastic gradient descent with momentum.
type SGD struct {
	Rate     float64 `json:"rate"`
	Momentum float64 `json:"momentum"`
	velocity map[int]float64
}

// NewSGD creates an SGD optimizer. A momentum of 0 disables it.
func NewSGD(rate, momentum float64) *SGD {
	return &SGD{Rate: rate, Momentum: momentum, velocity: map[int]float64{}}
}

// Kind implements Optimizer.
func (s *SGD) Kind() string { return "sgd" }

// Update implements Optimizer.
func (s *SGD) Update(key int, parameter, gradient float64) float64 {
	velocity := s.Momentum*s.velocity[key] - s.Rate*gradient
	s.velocity[key] = velocity
	return parameter + velocity
}

type sgdState struct {
	Rate     float64            `json:"rate"`
	Momentum float64            `json:"momentum"`
	Velocity map[string]float64 `json:"velocity"`
}

// MarshalJSON implements json.Marshaler.
func (s *SGD) MarshalJSON() ([]byte, error) {
	velocity := make(map[string]float64, len(s.velocity))
	for key, value := range s.velocity {
		velocity[strconv.Itoa(key)] = value
	}
	return json.Marshal(sgdState{Rate: s.Rate, Momentum: s.Momentum, Velocity: velocity})
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *SGD) UnmarshalJSON(data []byte) error {
	var state sgdState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	s.Rate = state.Rate
	s.Momentum = state.Momentum
	s.velocity = make(map[int]float64, len(state.Velocity))
	for key, value := range state.Velocity {
		v, _ := strconv.Atoi(key)
		s.velocity[v] = value
	}
	return nil
}

// RMSProp is an adaptive optimizer with weight decay and gradient clipping,
// mirroring the reference implementation in backprop-neat-js.
type RMSProp struct {
	Rate        float64 `json:"rate"`
	Decay       float64 `json:"decay"`
	Epsilon     float64 `json:"epsilon"`
	WeightDecay float64 `json:"weight_decay"`
	Clip        float64 `json:"clip"`
	squareAvg   map[int]float64
}

// NewRMSProp creates an RMSProp optimizer with sensible defaults for the
// omitted values.
func NewRMSProp(rate, decay, epsilon, weightDecay, clip float64) *RMSProp {
	if decay <= 0 || decay >= 1 {
		decay = 0.9
	}
	if epsilon <= 0 {
		epsilon = 1e-8
	}
	return &RMSProp{
		Rate:        rate,
		Decay:       decay,
		Epsilon:     epsilon,
		WeightDecay: weightDecay,
		Clip:        clip,
		squareAvg:   map[int]float64{},
	}
}

// Kind implements Optimizer.
func (r *RMSProp) Kind() string { return "rmsprop" }

// Update implements Optimizer.
func (r *RMSProp) Update(key int, parameter, gradient float64) float64 {
	if r.Clip > 0 {
		if gradient > r.Clip {
			gradient = r.Clip
		} else if gradient < -r.Clip {
			gradient = -r.Clip
		}
	}
	if r.WeightDecay > 0 {
		gradient += r.WeightDecay * parameter
	}
	average := r.Decay*r.squareAvg[key] + (1-r.Decay)*gradient*gradient
	r.squareAvg[key] = average
	return parameter - r.Rate*gradient/(math.Sqrt(average)+r.Epsilon)
}

type rmsPropState struct {
	Rate        float64            `json:"rate"`
	Decay       float64            `json:"decay"`
	Epsilon     float64            `json:"epsilon"`
	WeightDecay float64            `json:"weight_decay"`
	Clip        float64            `json:"clip"`
	SquareAvg   map[string]float64 `json:"square_avg"`
}

// MarshalJSON implements json.Marshaler.
func (r *RMSProp) MarshalJSON() ([]byte, error) {
	squareAvg := make(map[string]float64, len(r.squareAvg))
	for key, value := range r.squareAvg {
		squareAvg[strconv.Itoa(key)] = value
	}
	return json.Marshal(rmsPropState{
		Rate:        r.Rate,
		Decay:       r.Decay,
		Epsilon:     r.Epsilon,
		WeightDecay: r.WeightDecay,
		Clip:        r.Clip,
		SquareAvg:   squareAvg,
	})
}

// UnmarshalJSON implements json.Unmarshaler.
func (r *RMSProp) UnmarshalJSON(data []byte) error {
	var state rmsPropState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	r.Rate = state.Rate
	r.Decay = state.Decay
	r.Epsilon = state.Epsilon
	r.WeightDecay = state.WeightDecay
	r.Clip = state.Clip
	r.squareAvg = make(map[int]float64, len(state.SquareAvg))
	for key, value := range state.SquareAvg {
		v, _ := strconv.Atoi(key)
		r.squareAvg[v] = value
	}
	return nil
}
