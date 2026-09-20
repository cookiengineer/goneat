package neat

import (
	"sort"

	"github.com/cookiengineer/goneat/engine/random"
)

// Crossover produces a child from two parents using innovation alignment.
// Matching genes are inherited from either parent at random; disjoint and
// excess genes are inherited only from the fitter parent. A gene stays
// disabled only when it is disabled in both parents.
func Crossover(parentA, parentB *Genome, rng *random.Rand) *Genome {
	fitter, other := parentA, parentB
	if parentB.Fitness > parentA.Fitness {
		fitter, other = parentB, parentA
	}

	fitterGenes := make(map[int]ConnectionGene, len(fitter.Connections))
	for _, connection := range fitter.Connections {
		fitterGenes[connection.Innovation] = connection
	}
	otherGenes := make(map[int]ConnectionGene, len(other.Connections))
	for _, connection := range other.Connections {
		otherGenes[connection.Innovation] = connection
	}

	innovations := make([]int, 0, len(fitterGenes)+len(otherGenes))
	for innovation := range fitterGenes {
		innovations = append(innovations, innovation)
	}
	for innovation := range otherGenes {
		if _, ok := fitterGenes[innovation]; !ok {
			innovations = append(innovations, innovation)
		}
	}
	sort.Ints(innovations)

	child := &Genome{
		ID:         -1,
		Nodes:      append([]NodeGene(nil), fitter.Nodes...),
		Fitness:    0,
		Generation: fitter.Generation + 1,
		Parents:    []int{fitter.ID, other.ID},
	}

	for _, innovation := range innovations {
		gene, inFitter := fitterGenes[innovation]
		if !inFitter {
			// Disjoint/excess gene from the less fit parent is discarded.
			continue
		}
		if partner, inOther := otherGenes[innovation]; inOther {
			if rng.Float64() < 0.5 {
				gene.Weight = partner.Weight
			}
			gene.Enabled = gene.Enabled || partner.Enabled
		}
		child.Connections = append(child.Connections, gene)
	}

	// Import any node referenced by an inherited connection that came from
	// the other parent's topology.
	referenced := make(map[int]bool)
	for _, connection := range child.Connections {
		referenced[connection.In] = true
		referenced[connection.Out] = true
	}
	for id := range referenced {
		if !child.HasNode(id) {
			if node := other.NodeByID(id); node != nil {
				child.Nodes = append(child.Nodes, *node)
			} else if node := fitter.NodeByID(id); node != nil {
				child.Nodes = append(child.Nodes, *node)
			}
		}
	}

	sort.Slice(child.Nodes, func(i, j int) bool { return child.Nodes[i].ID < child.Nodes[j].ID })
	sort.Slice(child.Connections, func(i, j int) bool {
		return child.Connections[i].Innovation < child.Connections[j].Innovation
	})
	// Combining two acyclic parents can form a cycle; keep the child
	// feed-forward so it can be evaluated and differentiated.
	BreakCycles(child)
	return child
}
