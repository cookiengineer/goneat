package neat

import "fmt"

// Options configures genome construction, mutation, speciation and
// reproduction. The zero value is not usable; call DefaultOptions.
type Options struct {
	InputCount  int
	OutputCount int

	// PopulationSize is the number of genomes per generation.
	PopulationSize int
	// Seed makes the whole run reproducible.
	Seed int64

	// InitialConnectionRate is the probability of wiring a base connection
	// (input or bias to output) when a genome is created.
	InitialConnectionRate float64
	// InitialWeightStdDev is the standard deviation of initial weights.
	InitialWeightStdDev float64
	// OutputActivation is the activation used for output nodes.
	OutputActivation ActivationType
	// ActivationOptions is the pool of activations for new hidden nodes.
	ActivationOptions []ActivationType

	// WeightPerturbRate is the per-connection probability of adding gaussian
	// noise to a weight during mutation.
	WeightPerturbRate float64
	// WeightResetRate is the per-connection probability of replacing a weight
	// with a fresh random value during mutation.
	WeightResetRate float64
	// WeightPerturbPower is the standard deviation of weight noise.
	WeightPerturbPower float64

	// AddLinkRate is the probability of adding a connection during mutation.
	AddLinkRate float64
	// AddNodeRate is the probability of splitting a connection during mutation.
	AddNodeRate float64
	// ToggleEnableRate is the probability of toggling a connection's enabled
	// state during mutation.
	ToggleEnableRate float64

	// PruneLinkRate is the probability of removing an enabled connection
	// during a prune phase.
	PruneLinkRate float64
	// PruneNodeRate is the probability of removing a hidden node during a
	// prune phase.
	PruneNodeRate float64

	// Phases configures phased searching (complexify / simplify / backprop).
	Phases PhaseOptions

	// CompatibilityThreshold is the initial speciation distance threshold.
	CompatibilityThreshold float64
	// CompatibilityModifier is added/subtracted from the threshold to steer the
	// species count towards TargetSpeciesCount.
	CompatibilityModifier float64
	// TargetSpeciesCount is the species count the threshold tries to maintain.
	// Zero disables dynamic adjustment.
	TargetSpeciesCount int

	// CompatibilityExcessCoefficient weights excess genes (c1).
	CompatibilityExcessCoefficient float64
	// CompatibilityDisjointCoefficient weights disjoint genes (c2).
	CompatibilityDisjointCoefficient float64
	// CompatibilityWeightCoefficient weights average weight differences (c3).
	CompatibilityWeightCoefficient float64

	// Elitism is the number of best genomes copied unchanged into the next
	// generation.
	Elitism int
	// StagnationLimit is the number of generations a species may not improve
	// before it is removed.
	StagnationLimit int
	// SurvivalRate is the fraction of each species kept for reproduction.
	SurvivalRate float64
}

// DefaultOptions returns sensible defaults for the given input/output counts.
func DefaultOptions(inputCount, outputCount int) Options {
	return Options{
		InputCount:  inputCount,
		OutputCount: outputCount,

		PopulationSize: 150,
		Seed:           1,

		InitialConnectionRate: 1.0,
		InitialWeightStdDev:   1.0,
		OutputActivation:      ActivationSigmoid,
		ActivationOptions:     DefaultActivations(),

		WeightPerturbRate:  0.8,
		WeightResetRate:    0.1,
		WeightPerturbPower: 0.5,

		AddLinkRate:      0.08,
		AddNodeRate:      0.03,
		ToggleEnableRate: 0.01,

		PruneLinkRate: 0.10,
		PruneNodeRate: 0.03,

		Phases: defaultPhaseOptions(),

		CompatibilityThreshold: 3.0,
		CompatibilityModifier:  0.3,
		TargetSpeciesCount:     0,

		CompatibilityExcessCoefficient:   1.0,
		CompatibilityDisjointCoefficient: 1.0,
		CompatibilityWeightCoefficient:   0.4,

		Elitism:         1,
		StagnationLimit: 15,
		SurvivalRate:    0.2,
	}
}

// Validate returns an error if the options cannot produce a valid population.
func (o Options) Validate() error {
	if o.InputCount < 0 || o.OutputCount <= 0 {
		return fmt.Errorf("neat: invalid topology (%d inputs, %d outputs)", o.InputCount, o.OutputCount)
	}
	if o.PopulationSize < 2 {
		return fmt.Errorf("neat: population size must be at least 2, got %d", o.PopulationSize)
	}
	if len(o.ActivationOptions) == 0 {
		return fmt.Errorf("neat: at least one activation option is required")
	}
	if o.CompatibilityExcessCoefficient < 0 ||
		o.CompatibilityDisjointCoefficient < 0 ||
		o.CompatibilityWeightCoefficient < 0 {
		return fmt.Errorf("neat: compatibility coefficients must be non-negative")
	}
	return nil
}
