package neat

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/cookiengineer/goneat/engine/random"
)

const (
	populationFormat  = "goneat.neat.population"
	populationVersion = 1
)

type speciesState struct {
	ID               int     `json:"id"`
	RepresentativeID int     `json:"representative_id"`
	MemberIDs        []int   `json:"member_ids"`
	BestFitness      float64 `json:"best_fitness"`
	AverageFitness   float64 `json:"average_fitness"`
	Staleness        int     `json:"staleness"`
}

type populationState struct {
	Format        string          `json:"format"`
	Version       int             `json:"version"`
	Options       Options         `json:"options"`
	Generation    int             `json:"generation"`
	NextGenomeID  int             `json:"next_genome_id"`
	NextSpeciesID int             `json:"next_species_id"`
	Evaluations   int             `json:"evaluations"`
	Tracker       innovationState `json:"tracker"`
	RNG           json.RawMessage `json:"rng"`
	Genomes       []*Genome       `json:"genomes"`
	Species       []speciesState  `json:"species"`
	BestEver      *Genome         `json:"best_ever"`
}

// MarshalJSON serializes the population, including RNG and innovation state,
// so it can be resumed exactly.
func (p *Population) MarshalJSON() ([]byte, error) {
	rng, err := json.Marshal(p.rng)
	if err != nil {
		return nil, fmt.Errorf("neat: marshal rng: %w", err)
	}

	state := populationState{
		Format:        populationFormat,
		Version:       populationVersion,
		Options:       p.options,
		Generation:    p.generation,
		NextGenomeID:  p.nextGenomeID,
		NextSpeciesID: p.nextSpeciesID,
		Evaluations:   p.evaluations,
		Tracker:       p.tracker.MarshalState(),
		RNG:           rng,
		Genomes:       p.genomes,
		BestEver:      p.bestEver,
	}
	for _, species := range p.species {
		entry := speciesState{
			ID:             species.ID,
			BestFitness:    species.BestFitness,
			AverageFitness: species.AverageFitness,
			Staleness:      species.Staleness,
		}
		if species.Representative != nil {
			entry.RepresentativeID = species.Representative.ID
		}
		for _, member := range species.Members {
			entry.MemberIDs = append(entry.MemberIDs, member.ID)
		}
		state.Species = append(state.Species, entry)
	}

	return json.Marshal(state)
}

// UnmarshalJSON restores a population previously captured by MarshalJSON.
func (p *Population) UnmarshalJSON(data []byte) error {
	var state populationState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	if state.Format != populationFormat {
		return fmt.Errorf("neat: unexpected population format %q", state.Format)
	}
	if state.Version != populationVersion {
		return fmt.Errorf("neat: unsupported population version %d", state.Version)
	}
	if err := state.Options.Validate(); err != nil {
		return err
	}

	p.options = state.Options
	p.generation = state.Generation
	p.nextGenomeID = state.NextGenomeID
	p.nextSpeciesID = state.NextSpeciesID
	p.evaluations = state.Evaluations
	p.phases = newPhaseController(state.Options.Phases)
	p.tracker = NewInnovationTracker(state.Options.InputCount, state.Options.OutputCount)
	p.tracker.UnmarshalState(state.Tracker)

	p.rng = random.New(0)
	if len(state.RNG) > 0 {
		if err := json.Unmarshal(state.RNG, p.rng); err != nil {
			return fmt.Errorf("neat: unmarshal rng: %w", err)
		}
	}

	p.genomes = state.Genomes
	byID := make(map[int]*Genome, len(p.genomes))
	for _, genome := range p.genomes {
		byID[genome.ID] = genome
	}

	p.bestEver = state.BestEver
	if p.bestEver == nil && len(p.genomes) > 0 {
		p.bestEver = p.genomes[0].Clone()
		p.bestEver.Fitness = 0
	}

	p.species = p.species[:0]
	for _, entry := range state.Species {
		species := &Species{
			ID:             entry.ID,
			BestFitness:    entry.BestFitness,
			AverageFitness: entry.AverageFitness,
			Staleness:      entry.Staleness,
		}
		if representative, ok := byID[entry.RepresentativeID]; ok {
			species.Representative = representative
		}
		for _, memberID := range entry.MemberIDs {
			if member, ok := byID[memberID]; ok {
				species.Members = append(species.Members, member)
			}
		}
		p.species = append(p.species, species)
	}
	return nil
}

// Encode returns the JSON snapshot of the population.
func (p *Population) Encode() ([]byte, error) {
	return json.Marshal(p)
}

// Decode restores a population from a JSON snapshot.
func DecodePopulation(data []byte) (*Population, error) {
	if len(data) == 0 {
		return nil, errors.New("neat: empty population snapshot")
	}
	population := &Population{}
	if err := json.Unmarshal(data, population); err != nil {
		return nil, err
	}
	return population, nil
}
