package neat

import (
	"math"
	"sort"
)

// CompatibilityDistance measures how different two genomes are. It follows the
// NEAT formulation:
//
//	delta = c1*E/N + c2*D/N + c3*W
//
// where E is the number of excess genes, D the number of disjoint genes and W
// the average weight difference of matching genes. N is the larger genome
// size, or 1 for genomes with fewer than 20 genes to keep early distances from
// being over-diluted.
func CompatibilityDistance(a, b *Genome, excessCoefficient, disjointCoefficient, weightCoefficient float64) float64 {
	genesA := make(map[int]ConnectionGene, len(a.Connections))
	for _, connection := range a.Connections {
		genesA[connection.Innovation] = connection
	}
	genesB := make(map[int]ConnectionGene, len(b.Connections))
	for _, connection := range b.Connections {
		genesB[connection.Innovation] = connection
	}

	maxA := maxInnovation(genesA)
	maxB := maxInnovation(genesB)

	// Iterate in sorted innovation order so the accumulated weight difference
	// is independent of Go's randomized map iteration. This is required for
	// reproducible runs.
	innovations := make([]int, 0, len(genesA)+len(genesB))
	for innovation := range genesA {
		innovations = append(innovations, innovation)
	}
	for innovation := range genesB {
		if _, ok := genesA[innovation]; !ok {
			innovations = append(innovations, innovation)
		}
	}
	sort.Ints(innovations)

	var matching, disjoint, excess int
	var weightDifference float64

	for _, innovation := range innovations {
		geneA, inA := genesA[innovation]
		_, inB := genesB[innovation]
		switch {
		case inA && inB:
			matching++
			weightDifference += math.Abs(geneA.Weight - genesB[innovation].Weight)
		case inA:
			if innovation > maxB {
				excess++
			} else {
				disjoint++
			}
		default:
			if innovation > maxA {
				excess++
			} else {
				disjoint++
			}
		}
	}

	size := len(a.Connections)
	if len(b.Connections) > size {
		size = len(b.Connections)
	}
	if size < 20 {
		size = 1
	}

	distance := excessCoefficient*float64(excess)/float64(size) +
		disjointCoefficient*float64(disjoint)/float64(size)
	if matching > 0 {
		distance += weightCoefficient * weightDifference / float64(matching)
	}
	return distance
}

func maxInnovation(genes map[int]ConnectionGene) int {
	maximum := 0
	for innovation := range genes {
		if innovation > maximum {
			maximum = innovation
		}
	}
	return maximum
}
