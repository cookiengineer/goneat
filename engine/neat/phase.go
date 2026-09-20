package neat

// Phase identifies a search mode used by Phased Searching.
//
// Phased Searching (Gauci & Stanley) normally alternates between a
// complexifying phase that adds structure and a simplifying phase that removes
// it. goneat generalises this into four modes so gradient training can
// participate:
//
//   - PhaseDefault   complexify: add links and nodes (standard NEAT mutation)
//   - PhasePrune     simplify: remove links and nodes
//   - PhaseBackprop  exploit: gradient-train weights, no structural change
//   - PhaseHybrid    complexify and gradient-train (explore + exploit)
type Phase uint8

const (
	PhaseDefault Phase = iota
	PhasePrune
	PhaseBackprop
	PhaseHybrid
)

// String returns a human-readable phase name.
func (p Phase) String() string {
	switch p {
	case PhaseDefault:
		return "default"
	case PhasePrune:
		return "prune"
	case PhaseBackprop:
		return "backprop"
	case PhaseHybrid:
		return "hybrid"
	default:
		return "unknown"
	}
}

// Trainable reports whether the phase applies gradient training.
func (p Phase) Trainable() bool {
	return p == PhaseBackprop || p == PhaseHybrid
}

// PhaseTrainer refines a genome's weights in place. It is injected into the
// population so that engine/neat does not depend on engine/backprop.
type PhaseTrainer interface {
	Train(genome *Genome, iterations int) error
}

// PhaseTrainerFunc adapts a function to the PhaseTrainer interface.
type PhaseTrainerFunc func(genome *Genome, iterations int) error

// Train implements PhaseTrainer.
func (f PhaseTrainerFunc) Train(genome *Genome, iterations int) error {
	return f(genome, iterations)
}

// PhaseOptions configures phased searching.
type PhaseOptions struct {
	// Enabled switches phased searching on. When disabled the population stays
	// in Mode for the whole run.
	Enabled bool
	// Mode is the fixed phase used when Schedule is empty or Enabled is false.
	Mode Phase
	// Schedule cycles through the given phases when Enabled. An empty schedule
	// with Enabled means {PhaseDefault, PhasePrune}, the classic phased search.
	Schedule []Phase
	// Duration is the number of generations spent in each phase before
	// advancing. Zero keeps the phase until StallLimit triggers a switch.
	Duration int
	// StallLimit advances the phase early when the best fitness has not
	// improved for this many consecutive generations. Zero disables it.
	StallLimit int
	// TrainIterations is passed to Trainer on every offspring during a
	// trainable phase.
	TrainIterations int
	// Trainer performs the gradient step for PhaseBackprop and PhaseHybrid.
	// It is not serializable and must be re-injected after restoring a
	// population snapshot.
	Trainer PhaseTrainer `json:"-"`
}

// defaultPhaseOptions returns the standard phased-searching configuration.
func defaultPhaseOptions() PhaseOptions {
	return PhaseOptions{
		Enabled:         false,
		Mode:            PhaseDefault,
		Duration:        10,
		StallLimit:      8,
		TrainIterations: 20,
	}
}

// phaseController tracks the active phase and decides when to advance.
type phaseController struct {
	schedule    []Phase
	index       int
	duration    int
	stallLimit  int
	elapsed     int
	bestFitness float64
	haveBest    bool
	stalls      int
}

func newPhaseController(options PhaseOptions) *phaseController {
	schedule := options.Schedule
	if len(schedule) == 0 {
		if options.Enabled {
			schedule = []Phase{PhaseDefault, PhasePrune}
		} else {
			schedule = []Phase{options.Mode}
		}
	}
	return &phaseController{
		schedule:   append([]Phase(nil), schedule...),
		duration:   options.Duration,
		stallLimit: options.StallLimit,
	}
}

// current returns the active phase.
func (c *phaseController) current() Phase {
	if len(c.schedule) == 0 {
		return PhaseDefault
	}
	return c.schedule[c.index%len(c.schedule)]
}

// observe records the generation's best fitness and advances the phase when
// its duration elapses or fitness has stalled.
func (c *phaseController) observe(bestFitness float64) {
	c.elapsed++

	if !c.haveBest {
		c.bestFitness = bestFitness
		c.haveBest = true
	} else if bestFitness > c.bestFitness+1e-12 {
		c.bestFitness = bestFitness
		c.stalls = 0
	} else {
		c.stalls++
	}

	advance := false
	if c.duration > 0 && c.elapsed >= c.duration {
		advance = true
	}
	if c.stallLimit > 0 && c.stalls >= c.stallLimit {
		advance = true
	}
	if advance {
		c.advance()
	}
}

func (c *phaseController) advance() {
	if len(c.schedule) == 0 {
		return
	}
	c.index = (c.index + 1) % len(c.schedule)
	c.elapsed = 0
	c.stalls = 0
	c.haveBest = false
}
