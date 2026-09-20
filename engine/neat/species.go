package neat

import "sort"

// Species groups genomes that are structurally similar. Grouping protects
// innovation during the early generations and lets each niche optimize
// separately.
type Species struct {
	ID             int
	Representative *Genome
	Members        []*Genome

	BestFitness     float64
	AverageFitness  float64
	AdjustedFitness float64
	Staleness       int
	OffspringCount  int
	Stagnant        bool
}

func newSpecies(id int, representative *Genome) *Species {
	return &Species{
		ID:             id,
		Representative: representative,
		BestFitness:    representative.Fitness,
	}
}

// reset clears per-generation membership while keeping the representative.
func (s *Species) reset() {
	s.Members = s.Members[:0]
	s.BestFitness = 0
	s.AverageFitness = 0
	s.AdjustedFitness = 0
	s.OffspringCount = 0
	s.Stagnant = false
}

func (s *Species) add(genome *Genome) {
	s.Members = append(s.Members, genome)
	genome.SpeciesID = s.ID
}

// Champion returns the fittest member after sorting.
func (s *Species) Champion() *Genome {
	if len(s.Members) == 0 {
		return nil
	}
	return s.Members[0]
}

// sortByFitness orders members from fittest to least fit.
func (s *Species) sortByFitness() {
	sort.SliceStable(s.Members, func(i, j int) bool {
		return s.Members[i].Fitness > s.Members[j].Fitness
	})
}
