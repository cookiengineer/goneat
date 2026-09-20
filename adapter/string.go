package adapter

import (
	"fmt"
	"strings"
)

// StringAnchor ties a known string to a normalized position on [-1, 1].
type StringAnchor struct {
	Value    string  `json:"value"`
	Position float64 `json:"position"`
}

// StringAdapter vectorizes arbitrary strings against a dictionary of anchor
// strings. Encoding produces a weighted average of the anchor positions,
// weighted by string similarity; decoding returns the anchor nearest to the
// supplied position. The mapping is intentionally lossy: strings are projected
// onto a single similarity axis.
type StringAdapter struct {
	name         string
	anchors      []StringAnchor
	value        string
	placement    Point
	hasPlacement bool
}

// NewString creates a string adapter from anchors. At least one anchor is
// required.
func NewString(name string, anchors []StringAnchor) *StringAdapter {
	return &StringAdapter{
		name:    name,
		anchors: append([]StringAnchor(nil), anchors...),
	}
}

// WithPlacement assigns a substrate placement coordinate.
func (s *StringAdapter) WithPlacement(point Point) *StringAdapter {
	s.placement = point
	s.hasPlacement = true
	return s
}

// Name returns the adapter name.
func (s *StringAdapter) Name() string { return s.name }

// Anchors returns a copy of the configured anchors.
func (s *StringAdapter) Anchors() []StringAnchor { return append([]StringAnchor(nil), s.anchors...) }

// Placement implements Placed.
func (s *StringAdapter) Placement() (Point, bool) { return s.placement, s.hasPlacement }

// Dimensions is always 1.
func (s *StringAdapter) Dimensions() int { return 1 }

// SetValue sets the current string.
func (s *StringAdapter) SetValue(value string) { s.value = value }

// Value returns the current string.
func (s *StringAdapter) Value() string { return s.value }

// Sense projects the current string onto the similarity axis.
func (s *StringAdapter) Sense() []float64 {
	if len(s.anchors) == 0 {
		return []float64{0}
	}

	var weightedSum, weightTotal, nearestPosition float64
	nearestSimilarity := -1.0
	for _, anchor := range s.anchors {
		similarity := Similarity(s.value, anchor.Value)
		if similarity > nearestSimilarity {
			nearestSimilarity = similarity
			nearestPosition = anchor.Position
		}
		weightedSum += similarity * anchor.Position
		weightTotal += similarity
	}
	if weightTotal > 0 {
		return []float64{clampUnit(weightedSum / weightTotal)}
	}
	return []float64{clampUnit(nearestPosition)}
}

// Control snaps the current string to the anchor nearest to the encoded
// position.
func (s *StringAdapter) Control(values []float64) error {
	if len(values) < 1 {
		return fmt.Errorf("adapter: string %q expects 1 value, got %d", s.name, len(values))
	}
	if len(s.anchors) == 0 {
		return fmt.Errorf("adapter: string %q has no anchors", s.name)
	}
	target := clampUnit(values[0])
	best := s.anchors[0]
	bestDistance := absFloat(target - best.Position)
	for _, anchor := range s.anchors[1:] {
		if distance := absFloat(target - anchor.Position); distance < bestDistance {
			best = anchor
			bestDistance = distance
		}
	}
	s.value = best.Value
	return nil
}

// Similarity returns a value in [0, 1] describing how similar two strings are,
// based on normalized Levenshtein distance (case-insensitive).
func Similarity(a, b string) float64 {
	left := []rune(strings.ToLower(a))
	right := []rune(strings.ToLower(b))
	if len(left) == 0 && len(right) == 0 {
		return 1
	}
	distance := levenshtein(left, right)
	longest := len(left)
	if len(right) > longest {
		longest = len(right)
	}
	return 1 - float64(distance)/float64(longest)
}

func levenshtein(left, right []rune) int {
	if len(left) == 0 {
		return len(right)
	}
	if len(right) == 0 {
		return len(left)
	}

	previous := make([]int, len(right)+1)
	current := make([]int, len(right)+1)
	for j := range previous {
		previous[j] = j
	}

	for i := 1; i <= len(left); i++ {
		current[0] = i
		for j := 1; j <= len(right); j++ {
			cost := 1
			if left[i-1] == right[j-1] {
				cost = 0
			}
			current[j] = minInt(
				current[j-1]+1,
				minInt(previous[j]+1, previous[j-1]+cost),
			)
		}
		previous, current = current, previous
	}
	return previous[len(right)]
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func absFloat(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}
