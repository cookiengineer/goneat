package vector

import (
	"math"
	"math/rand"
	"testing"
)

const relativeTolerance = 1e-9

// almostEqual compares floating point results with a relative tolerance, since
// the scalar and SIMD backends accumulate in different orders.
func almostEqual(got, want float64) bool {
	scale := math.Max(1, math.Max(math.Abs(got), math.Abs(want)))
	return math.Abs(got-want) <= relativeTolerance*scale
}

func naiveDot(a, b []float64) float64 {
	n := min(len(a), len(b))
	var sum float64
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func naiveSum(a []float64) float64 {
	var sum float64
	for _, v := range a {
		sum += v
	}
	return sum
}

func randomSlice(rng *rand.Rand, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.Float64()*200 - 100
	}
	return out
}

// testLengths exercises zero, sub-lane, exactly-lane and awkward tail sizes so
// both the scalar and the SIMD backend are covered regardless of lane width.
func testLengths() []int {
	return []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31, 33, 64, 100, 129}
}

func TestDotMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range testLengths() {
		a := randomSlice(rng, n)
		b := randomSlice(rng, n)
		if got, want := Dot(a, b), naiveDot(a, b); !almostEqual(got, want) {
			t.Fatalf("Dot(n=%d) = %v, want %v", n, got, want)
		}
	}
}

func TestAddMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range testLengths() {
		a := randomSlice(rng, n)
		b := randomSlice(rng, n)
		got := make([]float64, n)
		Add(got, a, b)
		for i := 0; i < n; i++ {
			if want := a[i] + b[i]; !almostEqual(got[i], want) {
				t.Fatalf("Add(n=%d) index %d = %v, want %v", n, i, got[i], want)
			}
		}
	}
}

func TestSubMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range testLengths() {
		a := randomSlice(rng, n)
		b := randomSlice(rng, n)
		got := make([]float64, n)
		Sub(got, a, b)
		for i := 0; i < n; i++ {
			if want := a[i] - b[i]; !almostEqual(got[i], want) {
				t.Fatalf("Sub(n=%d) index %d = %v, want %v", n, i, got[i], want)
			}
		}
	}
}

func TestScaleMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, n := range testLengths() {
		a := randomSlice(rng, n)
		got := make([]float64, n)
		Scale(got, a, 2.5)
		for i := 0; i < n; i++ {
			if want := a[i] * 2.5; !almostEqual(got[i], want) {
				t.Fatalf("Scale(n=%d) index %d = %v, want %v", n, i, got[i], want)
			}
		}
	}
}

func TestMulAddMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, n := range testLengths() {
		dst := randomSlice(rng, n)
		a := randomSlice(rng, n)
		b := randomSlice(rng, n)
		want := make([]float64, n)
		for i := 0; i < n; i++ {
			want[i] = dst[i] + a[i]*b[i]*0.25
		}
		MulAdd(dst, a, b, 0.25)
		for i := 0; i < n; i++ {
			if !almostEqual(dst[i], want[i]) {
				t.Fatalf("MulAdd(n=%d) index %d = %v, want %v", n, i, dst[i], want[i])
			}
		}
	}
}

func TestSumMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	for _, n := range testLengths() {
		a := randomSlice(rng, n)
		if got, want := Sum(a), naiveSum(a); !almostEqual(got, want) {
			t.Fatalf("Sum(n=%d) = %v, want %v", n, got, want)
		}
	}
}

func TestShortSliceSemantics(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5}
	if got, want := Dot(a, b), 1*4.0+2*5.0; got != want {
		t.Fatalf("Dot shorter = %v, want %v", got, want)
	}
	dst := make([]float64, 2)
	Add(dst, a, b)
	if dst[0] != 5 || dst[1] != 7 {
		t.Fatalf("Add shorter = %v", dst)
	}
}
