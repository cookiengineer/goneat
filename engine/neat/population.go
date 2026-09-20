package neat

import (
	"errors"
	"math"
	"sort"

	"github.com/cookiengineer/goneat/engine/random"
)

// Evaluator assigns a fitness to a genome. Higher is better.
type Evaluator interface {
	Evaluate(genome *Genome) (float64, error)
}

// EvaluatorFunc adapts a function to the Evaluator interface.
type EvaluatorFunc func(genome *Genome) (float64, error)

// Evaluate implements Evaluator.
func (f EvaluatorFunc) Evaluate(genome *Genome) (float64, error) { return f(genome) }

// GenerationStats summarizes one completed generation.
type GenerationStats struct {
	Generation       int     `json:"generation"`
	SpeciesCount     int     `json:"species_count"`
	PopulationSize   int     `json:"population_size"`
	BestFitness      float64 `json:"best_fitness"`
	AverageFitness   float64 `json:"average_fitness"`
	ChampionID       int     `json:"champion_id"`
	TotalEvaluations int     `json:"total_evaluations"`
}

// Observer is notified after each generation.
type Observer interface {
	GenerationEvaluated(stats GenerationStats)
}

// ObserverFunc adapts a function to the Observer interface.
type ObserverFunc func(stats GenerationStats)

// GenerationEvaluated implements Observer.
func (f ObserverFunc) GenerationEvaluated(stats GenerationStats) { f(stats) }

// Population owns a set of genomes, their species, the innovation tracker and
// the random source. All stochastic decisions flow through its RNG, so a
// serialized population resumes deterministically.
type Population struct {
	options Options

	genomes       []*Genome
	species       []*Species
	tracker       *InnovationTracker
	rng           *random.Rand
	phases        *phaseController
	generation    int
	nextGenomeID  int
	nextSpeciesID int
	bestEver      *Genome
	evaluations   int
}

// NewPopulation creates an initial population from the options.
func NewPopulation(options Options) (*Population, error) {
	if err := options.Validate(); err != nil {
		return nil, err
	}

	population := &Population{
		options:       options,
		tracker:       NewInnovationTracker(options.InputCount, options.OutputCount),
		rng:           random.New(options.Seed),
		phases:        newPhaseController(options.Phases),
		nextGenomeID:  1,
		nextSpeciesID: 1,
	}

	for i := 0; i < options.PopulationSize; i++ {
		genome := NewSeededGenome(options, population.tracker, population.rng)
		genome.ID = population.takeGenomeID()
		genome.Generation = 0
		population.genomes = append(population.genomes, genome)
	}
	population.bestEver = population.genomes[0].Clone()
	population.bestEver.Fitness = math.Inf(-1)
	return population, nil
}

func (p *Population) takeGenomeID() int {
	id := p.nextGenomeID
	p.nextGenomeID++
	return id
}

// Options returns the population options.
func (p *Population) Options() Options { return p.options }

// Generation returns the current generation index (starting at 1 after the
// first Epoch).
func (p *Population) Generation() int { return p.generation }

// Phase returns the active search phase.
func (p *Population) Phase() Phase { return p.phases.current() }

// Genomes returns the current genomes.
func (p *Population) Genomes() []*Genome { return p.genomes }

// Species returns the current species.
func (p *Population) Species() []*Species { return p.species }

// RNG exposes the random source (used by snapshots and tests).
func (p *Population) RNG() *random.Rand { return p.rng }

// Tracker exposes the innovation tracker.
func (p *Population) Tracker() *InnovationTracker { return p.tracker }

// BestGenome returns a copy of the fittest genome seen so far.
func (p *Population) BestGenome() *Genome {
	if p.bestEver == nil {
		return nil
	}
	return p.bestEver.Clone()
}

// Epoch evaluates every genome, speciates, and produces the next generation.
func (p *Population) Epoch(evaluator Evaluator, observer Observer) (GenerationStats, error) {
	if evaluator == nil {
		return GenerationStats{}, errors.New("neat: evaluator is nil")
	}

	// 1. Evaluate.
	bestFitness := math.Inf(-1)
	totalFitness := 0.0
	var champion *Genome
	for _, genome := range p.genomes {
		fitness, err := evaluator.Evaluate(genome)
		if err != nil {
			return GenerationStats{}, err
		}
		genome.Fitness = fitness
		p.evaluations++
		totalFitness += fitness
		if fitness > bestFitness {
			bestFitness = fitness
			champion = genome
		}
		if genome.Fitness > p.bestEver.Fitness {
			p.bestEver = genome.Clone()
		}
	}

	averageFitness := totalFitness / float64(len(p.genomes))

	// 2. Speciate and compute per-species statistics.
	p.speciate()
	p.updateSpeciesStatistics()

	// 3. Steer the species count and reproduce.
	p.adjustCompatibilityThreshold()
	if err := p.reproduce(); err != nil {
		return GenerationStats{}, err
	}
	p.generation++
	p.phases.observe(bestFitness)

	stats := GenerationStats{
		Generation:       p.generation,
		SpeciesCount:     len(p.species),
		PopulationSize:   len(p.genomes),
		BestFitness:      bestFitness,
		AverageFitness:   averageFitness,
		TotalEvaluations: p.evaluations,
	}
	if champion != nil {
		stats.ChampionID = champion.ID
	}
	if observer != nil {
		observer.GenerationEvaluated(stats)
	}
	return stats, nil
}

func (p *Population) speciate() {
	for _, species := range p.species {
		species.reset()
	}

	for _, genome := range p.genomes {
		placed := false
		for _, species := range p.species {
			if species.Representative == nil {
				continue
			}
			distance := CompatibilityDistance(
				genome, species.Representative,
				p.options.CompatibilityExcessCoefficient,
				p.options.CompatibilityDisjointCoefficient,
				p.options.CompatibilityWeightCoefficient,
			)
			if distance < p.options.CompatibilityThreshold {
				species.add(genome)
				placed = true
				break
			}
		}
		if !placed {
			species := newSpecies(p.nextSpeciesID, genome)
			p.nextSpeciesID++
			species.add(genome)
			p.species = append(p.species, species)
		}
	}

	// Drop species that lost all members.
	kept := p.species[:0]
	for _, species := range p.species {
		if len(species.Members) > 0 {
			kept = append(kept, species)
		}
	}
	p.species = kept
}

func (p *Population) updateSpeciesStatistics() {
	for _, species := range p.species {
		species.sortByFitness()

		total := 0.0
		for _, member := range species.Members {
			total += member.Fitness
		}
		species.AverageFitness = total / float64(len(species.Members))

		currentBest := species.Members[0].Fitness
		if currentBest > species.BestFitness {
			species.BestFitness = currentBest
			species.Staleness = 0
		} else {
			species.Staleness++
		}
		species.Stagnant = p.options.StagnationLimit > 0 && species.Staleness > p.options.StagnationLimit
	}
}

func (p *Population) adjustCompatibilityThreshold() {
	if p.options.TargetSpeciesCount <= 0 {
		return
	}
	if len(p.species) < p.options.TargetSpeciesCount {
		p.options.CompatibilityThreshold -= p.options.CompatibilityModifier
	} else if len(p.species) > p.options.TargetSpeciesCount {
		p.options.CompatibilityThreshold += p.options.CompatibilityModifier
	}
	if p.options.CompatibilityThreshold < 0.1 {
		p.options.CompatibilityThreshold = 0.1
	}
}

func (p *Population) reproduce() error {
	// Fitness sharing: offset fitness to be positive, then divide by species
	// size so crowded species do not dominate.
	minimum := math.Inf(1)
	for _, genome := range p.genomes {
		if genome.Fitness < minimum {
			minimum = genome.Fitness
		}
	}
	offset := 0.0
	if minimum < 0 {
		offset = -minimum + 1e-9
	}

	for _, species := range p.species {
		size := float64(len(species.Members))
		total := 0.0
		for _, member := range species.Members {
			total += (member.Fitness + offset) / size
		}
		species.AdjustedFitness = total / size
	}

	p.allocateOffspring()

	next := make([]*Genome, 0, p.options.PopulationSize)
	for _, species := range p.species {
		if species.Stagnant || species.OffspringCount <= 0 {
			species.Members = nil
			continue
		}
		children := p.breedSpecies(species)
		next = append(next, children...)
		// Keep species pointing at current genomes so snapshots restore cleanly.
		if len(children) > 0 {
			species.Members = children
			if species.Representative == nil || !containsGenomeID(children, species.Representative.ID) {
				species.Representative = children[0]
			}
		}
	}

	// Guard against allocation shortfalls (e.g. all species stagnant).
	for len(next) < p.options.PopulationSize {
		source := p.bestEver
		if source == nil {
			source = p.genomes[p.rng.Intn(len(p.genomes))]
		}
		child := source.Clone()
		child.ID = p.takeGenomeID()
		child.Generation = p.generation + 1
		child.Fitness = 0
		MutateWithPhase(child, p.tracker, p.options, p.rng, p.phases.current())
		next = append(next, child)
	}
	if len(next) > p.options.PopulationSize {
		next = next[:p.options.PopulationSize]
	}

	p.trainOffspring(next)
	p.genomes = next
	return nil
}

// trainOffspring applies gradient training during trainable phases.
func (p *Population) trainOffspring(genomes []*Genome) {
	if !p.phases.current().Trainable() {
		return
	}
	trainer := p.options.Phases.Trainer
	if trainer == nil {
		return
	}
	iterations := p.options.Phases.TrainIterations
	if iterations < 1 {
		iterations = 1
	}
	for _, genome := range genomes {
		_ = trainer.Train(genome, iterations)
	}
}

// allocateOffspring distributes the population budget across non-stagnant
// species using explicit fitness sharing and the largest-remainder method.
func (p *Population) allocateOffspring() {
	total := p.options.PopulationSize

	var eligible []*Species
	for _, species := range p.species {
		if !species.Stagnant && species.AdjustedFitness > 0 {
			eligible = append(eligible, species)
		}
	}
	if len(eligible) == 0 {
		for _, species := range p.species {
			species.OffspringCount = 0
		}
		return
	}

	for _, species := range p.species {
		species.OffspringCount = 0
	}

	// Give every eligible species at least one slot when possible.
	remaining := total
	if remaining >= len(eligible) {
		for _, species := range eligible {
			species.OffspringCount = 1
			remaining--
		}
	}

	if remaining > 0 {
		totalWeight := 0.0
		for _, species := range eligible {
			totalWeight += species.AdjustedFitness
		}
		if totalWeight <= 0 {
			for index := 0; remaining > 0; index++ {
				eligible[index%len(eligible)].OffspringCount++
				remaining--
			}
		} else {
			type remainder struct {
				species *Species
				value   float64
			}
			remainders := make([]remainder, 0, len(eligible))
			assigned := 0
			for _, species := range eligible {
				exact := float64(remaining) * species.AdjustedFitness / totalWeight
				base := int(math.Floor(exact))
				species.OffspringCount += base
				assigned += base
				remainders = append(remainders, remainder{species: species, value: exact - math.Floor(exact)})
			}
			leftover := remaining - assigned
			sort.SliceStable(remainders, func(i, j int) bool {
				if remainders[i].value == remainders[j].value {
					return remainders[i].species.ID < remainders[j].species.ID
				}
				return remainders[i].value > remainders[j].value
			})
			for i := 0; i < leftover; i++ {
				remainders[i%len(remainders)].species.OffspringCount++
			}
		}
	}

	// Correct any drift caused by rounding so the budget is exact.
	sum := 0
	for _, species := range p.species {
		sum += species.OffspringCount
	}
	for sum != total && len(eligible) > 0 {
		if sum > total {
			worst := eligible[0]
			for _, species := range eligible {
				if species.OffspringCount > worst.OffspringCount {
					worst = species
				}
			}
			if worst.OffspringCount == 0 {
				break
			}
			worst.OffspringCount--
			sum--
		} else {
			best := eligible[0]
			for _, species := range eligible {
				if species.AdjustedFitness > best.AdjustedFitness {
					best = species
				}
			}
			best.OffspringCount++
			sum++
		}
	}
}

func (p *Population) breedSpecies(species *Species) []*Genome {
	count := species.OffspringCount
	if count <= 0 || len(species.Members) == 0 {
		return nil
	}

	species.sortByFitness()
	parentsCount := int(math.Ceil(float64(len(species.Members)) * p.options.SurvivalRate))
	if parentsCount < 2 {
		parentsCount = min(2, len(species.Members))
	}
	if parentsCount > len(species.Members) {
		parentsCount = len(species.Members)
	}
	parents := species.Members[:parentsCount]

	children := make([]*Genome, 0, count)
	if p.options.Elitism > 0 {
		elite := species.Champion().Clone()
		elite.ID = p.takeGenomeID()
		elite.Generation = p.generation + 1
		children = append(children, elite)
	}

	for len(children) < count {
		var child *Genome
		if len(parents) >= 2 {
			mother := parents[p.rng.Intn(len(parents))]
			father := parents[p.rng.Intn(len(parents))]
			child = Crossover(mother, father, p.rng)
		} else {
			child = parents[0].Clone()
		}
		child.ID = p.takeGenomeID()
		child.Generation = p.generation + 1
		child.Fitness = 0
		MutateWithPhase(child, p.tracker, p.options, p.rng, p.phases.current())
		children = append(children, child)
	}
	return children
}

// containsGenomeID reports whether the given ID is present in the slice.
func containsGenomeID(genomes []*Genome, id int) bool {
	for _, genome := range genomes {
		if genome.ID == id {
			return true
		}
	}
	return false
}
