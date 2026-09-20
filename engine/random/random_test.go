package random

import (
	"encoding/json"
	"math"
	"testing"
)

func TestDeterministicSequence(t *testing.T) {
	a := New(42)
	b := New(42)
	for i := 0; i < 100; i++ {
		if a.Uint64() != b.Uint64() {
			t.Fatalf("sequences diverged at %d", i)
		}
	}
}

func TestDifferentSeedsDiverge(t *testing.T) {
	a := New(1)
	b := New(2)
	same := 0
	for i := 0; i < 100; i++ {
		if a.Uint64() == b.Uint64() {
			same++
		}
	}
	if same > 1 {
		t.Fatalf("different seeds produced %d equal values", same)
	}
}

func TestFloat64Range(t *testing.T) {
	rng := New(7)
	for i := 0; i < 10000; i++ {
		value := rng.Float64()
		if value < 0 || value >= 1 {
			t.Fatalf("Float64 out of range: %v", value)
		}
	}
}

func TestIntnRange(t *testing.T) {
	rng := New(8)
	seen := map[int]bool{}
	for i := 0; i < 10000; i++ {
		value := rng.Intn(5)
		if value < 0 || value >= 5 {
			t.Fatalf("Intn out of range: %d", value)
		}
		seen[value] = true
	}
	if len(seen) != 5 {
		t.Fatalf("Intn only produced %v", seen)
	}
}

func TestNormFloat64Approx(t *testing.T) {
	rng := New(9)
	const samples = 200000
	var sum float64
	for i := 0; i < samples; i++ {
		sum += rng.NormFloat64()
	}
	mean := sum / samples
	if math.Abs(mean) > 0.02 {
		t.Fatalf("normal mean = %v, want near 0", mean)
	}
}

func TestJSONRoundTripResumes(t *testing.T) {
	original := New(123)
	// Advance the stream a little, then snapshot.
	for i := 0; i < 17; i++ {
		original.Uint64()
	}
	state, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	restored := New(1)
	if err := json.Unmarshal(state, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	for i := 0; i < 100; i++ {
		if original.Uint64() != restored.Uint64() {
			t.Fatalf("restored stream diverged at %d", i)
		}
	}
}

func TestJSONRoundTripWithSpareGaussian(t *testing.T) {
	original := New(55)
	// Build up an odd number of gaussians so the spare slot is populated.
	original.NormFloat64()
	original.NormFloat64()
	original.NormFloat64()

	state, _ := json.Marshal(original)
	restored := New(1)
	if err := json.Unmarshal(state, restored); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	for i := 0; i < 50; i++ {
		if original.NormFloat64() != restored.NormFloat64() {
			t.Fatalf("gaussian stream diverged at %d", i)
		}
	}
}
