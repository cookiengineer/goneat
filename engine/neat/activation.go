package neat

import (
	"fmt"
	"math"
	"strings"
)

// ActivationType identifies the transfer function of a node.
type ActivationType uint8

const (
	// ActivationLinear is the identity function.
	ActivationLinear ActivationType = iota
	// ActivationSigmoid is the logistic function, range (0, 1).
	ActivationSigmoid
	// ActivationTanh is the hyperbolic tangent, range (-1, 1).
	ActivationTanh
	// ActivationReLU is the rectified linear unit.
	ActivationReLU
	// ActivationGaussian is exp(-x^2), range (0, 1].
	ActivationGaussian
	// ActivationSin is the sine function.
	ActivationSin
	// ActivationCos is the cosine function.
	ActivationCos
	// ActivationAbs is the absolute value.
	ActivationAbs
	// ActivationMult multiplies its weighted inputs instead of summing them.
	ActivationMult
	// ActivationAdd sums its weighted inputs without an extra transfer.
	ActivationAdd
)

var activationNames = map[ActivationType]string{
	ActivationLinear:   "linear",
	ActivationSigmoid:  "sigmoid",
	ActivationTanh:     "tanh",
	ActivationReLU:     "relu",
	ActivationGaussian: "gaussian",
	ActivationSin:      "sin",
	ActivationCos:      "cos",
	ActivationAbs:      "abs",
	ActivationMult:     "mult",
	ActivationAdd:      "add",
}

var activationByName = func() map[string]ActivationType {
	out := make(map[string]ActivationType, len(activationNames))
	for kind, name := range activationNames {
		out[name] = kind
	}
	return out
}()

// String returns the canonical lower-case name.
func (a ActivationType) String() string {
	if name, ok := activationNames[a]; ok {
		return name
	}
	return fmt.Sprintf("activation(%d)", uint8(a))
}

// MarshalText implements encoding.TextMarshaler for human-readable JSON.
func (a ActivationType) MarshalText() ([]byte, error) {
	return []byte(a.String()), nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (a *ActivationType) UnmarshalText(text []byte) error {
	parsed, err := ParseActivation(string(text))
	if err != nil {
		return err
	}
	*a = parsed
	return nil
}

// ParseActivation resolves an activation by name.
func ParseActivation(name string) (ActivationType, error) {
	if kind, ok := activationByName[strings.ToLower(strings.TrimSpace(name))]; ok {
		return kind, nil
	}
	return ActivationLinear, fmt.Errorf("neat: unknown activation %q", name)
}

// Apply evaluates the transfer function. Aggregation activations (Add, Mult)
// behave as the identity here; the network layer handles the aggregation.
func (a ActivationType) Apply(x float64) float64 {
	switch a {
	case ActivationSigmoid:
		return 1 / (1 + math.Exp(-x))
	case ActivationTanh:
		return math.Tanh(x)
	case ActivationReLU:
		if x > 0 {
			return x
		}
		return 0
	case ActivationGaussian:
		return math.Exp(-x * x)
	case ActivationSin:
		return math.Sin(x)
	case ActivationCos:
		return math.Cos(x)
	case ActivationAbs:
		return math.Abs(x)
	default:
		return x
	}
}

// Derivative returns d(Apply)/dx evaluated at the pre-activation x. It returns
// 0 for aggregations, which are differentiated specially by the backprop
// graph.
func (a ActivationType) Derivative(x float64) float64 {
	switch a {
	case ActivationSigmoid:
		sigmoid := 1 / (1 + math.Exp(-x))
		return sigmoid * (1 - sigmoid)
	case ActivationTanh:
		tanh := math.Tanh(x)
		return 1 - tanh*tanh
	case ActivationReLU:
		if x > 0 {
			return 1
		}
		return 0
	case ActivationGaussian:
		return -2 * x * math.Exp(-x*x)
	case ActivationSin:
		return math.Cos(x)
	case ActivationCos:
		return -math.Sin(x)
	case ActivationAbs:
		if x > 0 {
			return 1
		}
		if x < 0 {
			return -1
		}
		return 0
	case ActivationLinear:
		return 1
	default:
		return 0
	}
}

// IsAggregation reports whether the activation aggregates its weighted inputs
// instead of summing them and applying a transfer function.
func (a ActivationType) IsAggregation() bool {
	return a == ActivationMult || a == ActivationAdd
}

// DefaultActivations returns the activation set used for new hidden nodes.
func DefaultActivations() []ActivationType {
	return []ActivationType{
		ActivationSigmoid,
		ActivationTanh,
		ActivationReLU,
		ActivationGaussian,
		ActivationSin,
		ActivationAbs,
		ActivationMult,
		ActivationAdd,
	}
}
