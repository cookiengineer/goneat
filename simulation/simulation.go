package simulation

import (
	"context"
	"errors"
	"fmt"

	"github.com/cookiengineer/goneat/agent"
	"github.com/cookiengineer/goneat/engine/neat"
)

// Options configures a Simulation.
type Options struct {
	// Ticks is the maximum ticks per episode.
	Ticks int
	// DT is the timestep handed to World.Step.
	DT float64
	// Seed seeds the world reset schedule.
	Seed int64
	// Runner defaults to ChannelRunner when nil.
	Runner Runner
	// Reward and Punish are the user's per-tick callbacks.
	Reward RewardFunc
	Punish PunishFunc
	// Observer receives generation statistics.
	Observer neat.Observer
	// BrainFactory builds a brain for a genome. It defaults to NEATBrain.
	BrainFactory func(genome *neat.Genome) (agent.Brain, error)
	// AutoSavePath, when non-empty, writes a snapshot to this path every
	// AutoSaveEvery generations so partial training is persisted to disk.
	AutoSavePath string
	// AutoSaveEvery is the checkpoint interval in generations. A value <= 0
	// disables automatic snapshots.
	AutoSaveEvery int
}

// Simulation runs a population of agents against a World, generation by
// generation.
type Simulation struct {
	world        World
	population   *neat.Population
	runner       Runner
	reward       RewardFunc
	punish       PunishFunc
	observer     neat.Observer
	brainFactory func(genome *neat.Genome) (agent.Brain, error)

	ticks int
	dt    float64
	seed  int64

	autoSavePath  string
	autoSaveEvery int

	generation int
	lastStats  neat.GenerationStats
	history    []neat.GenerationStats
}

// NewSimulation validates the inputs and returns a ready simulation.
func NewSimulation(world World, population *neat.Population, options Options) (*Simulation, error) {
	if world == nil {
		return nil, errors.New("simulation: nil world")
	}
	if population == nil {
		return nil, errors.New("simulation: nil population")
	}
	if world.SensorDimensions() <= 0 || world.ControlDimensions() <= 0 {
		return nil, errors.New("simulation: world dimensions must be positive")
	}

	simulation := &Simulation{
		world:         world,
		population:    population,
		runner:        options.Runner,
		reward:        options.Reward,
		punish:        options.Punish,
		observer:      options.Observer,
		brainFactory:  options.BrainFactory,
		ticks:         options.Ticks,
		dt:            options.DT,
		seed:          options.Seed,
		autoSavePath:  options.AutoSavePath,
		autoSaveEvery: options.AutoSaveEvery,
	}
	if simulation.runner == nil {
		simulation.runner = ChannelRunner{}
	}
	if simulation.brainFactory == nil {
		simulation.brainFactory = func(genome *neat.Genome) (agent.Brain, error) {
			return agent.NewNEATBrain(genome)
		}
	}
	if simulation.ticks <= 0 {
		simulation.ticks = 1000
	}
	if simulation.dt <= 0 {
		simulation.dt = 1.0 / 60.0
	}
	return simulation, nil
}

// World returns the environment.
func (s *Simulation) World() World { return s.world }

// Population returns the genome population.
func (s *Simulation) Population() *neat.Population { return s.population }

// Generation returns the current generation.
func (s *Simulation) Generation() int { return s.generation }

// LastStats returns the most recent generation statistics.
func (s *Simulation) LastStats() neat.GenerationStats { return s.lastStats }

// History returns statistics for every completed generation.
func (s *Simulation) History() []neat.GenerationStats { return s.history }

// SetReward replaces the reward callback.
func (s *Simulation) SetReward(reward RewardFunc) { s.reward = reward }

// SetPunish replaces the punish callback.
func (s *Simulation) SetPunish(punish PunishFunc) { s.punish = punish }

// SetRunner replaces the runner.
func (s *Simulation) SetRunner(runner Runner) { s.runner = runner }

// SetObserver replaces the generation observer.
func (s *Simulation) SetObserver(observer neat.Observer) { s.observer = observer }

// SetAutoSave configures periodic snapshotting. A path of "" or an interval
// <= 0 disables it.
func (s *Simulation) SetAutoSave(path string, every int) {
	s.autoSavePath = path
	s.autoSaveEvery = every
}

// AutoSavePath returns the configured autosave path.
func (s *Simulation) AutoSavePath() string { return s.autoSavePath }

// RunEpisode builds one agent per genome and plays a single episode. Fitness
// accumulated by the callbacks stays on the genomes.
func (s *Simulation) RunEpisode(ctx context.Context) error {
	agents, err := s.buildAgents()
	if err != nil {
		return err
	}
	s.world.Reset(s.population.RNG().Int63())
	if err := s.runner.Run(ctx, s.world, agents, RunConfig{
		Ticks:  s.ticks,
		DT:     s.dt,
		Reward: s.reward,
		Punish: s.punish,
	}); err != nil {
		return err
	}

	// Fitness lives on the agent during an episode; transfer it to the genome,
	// which is the unit of evolution.
	genomes := s.population.Genomes()
	for index, current := range agents {
		if index < len(genomes) {
			genomes[index].Fitness = current.Fitness
		}
	}
	return nil
}

// RunGeneration plays an episode and then evolves the population.
func (s *Simulation) RunGeneration(ctx context.Context) (neat.GenerationStats, error) {
	if err := s.RunEpisode(ctx); err != nil {
		return neat.GenerationStats{}, err
	}
	stats, err := s.population.Epoch(alreadyEvaluated{}, s.observer)
	if err != nil {
		return neat.GenerationStats{}, err
	}
	s.generation = s.population.Generation()
	s.lastStats = stats
	s.history = append(s.history, stats)

	if s.autoSavePath != "" && s.autoSaveEvery > 0 && s.generation%s.autoSaveEvery == 0 {
		if err := s.SaveTo(s.autoSavePath); err != nil {
			return stats, fmt.Errorf("simulation: autosave: %w", err)
		}
	}
	return stats, nil
}

// RunGenerations runs RunGeneration count times or until the context is
// cancelled.
func (s *Simulation) RunGenerations(ctx context.Context, count int) ([]neat.GenerationStats, error) {
	stats := make([]neat.GenerationStats, 0, count)
	for i := 0; i < count; i++ {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		stat, err := s.RunGeneration(ctx)
		if err != nil {
			return stats, err
		}
		stats = append(stats, stat)
	}
	return stats, nil
}

func (s *Simulation) buildAgents() ([]*agent.Agent, error) {
	genomes := s.population.Genomes()
	agents := make([]*agent.Agent, len(genomes))
	for index, genome := range genomes {
		brain, err := s.brainFactory(genome)
		if err != nil {
			return nil, fmt.Errorf("simulation: build brain for genome %d: %w", genome.ID, err)
		}
		agents[index] = agent.NewAgent(fmt.Sprintf("genome-%d", genome.ID), brain)
	}
	return agents, nil
}

// alreadyEvaluated is used after a world episode has assigned fitness to every
// genome; it lets Population.Epoch perform only speciaton and reproduction.
type alreadyEvaluated struct{}

func (alreadyEvaluated) Evaluate(genome *neat.Genome) (float64, error) {
	return genome.Fitness, nil
}
