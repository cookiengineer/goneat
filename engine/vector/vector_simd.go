//go:build goexperiment.simd

package vector

import (
	"simd"
	"sync"
)

// vectorLength is the number of float64 lanes processed per SIMD iteration.
// It is determined once and is the same for every vector in the process (see
// the simd package documentation). Detection is deferred into a function:
// calling the bodyless simd functions from a package-level initializer does
// not link correctly.
var (
	vectorLengthOnce  sync.Once
	vectorLengthValue int
)

func vectorLength() int {
	vectorLengthOnce.Do(func() {
		vectorLengthValue = simd.BroadcastFloat64s(0).Len()
	})
	return vectorLengthValue
}

// horizontalSum returns the sum of all lanes of v.
func horizontalSum(v simd.Float64s, length int) float64 {
	var lanes [64]float64
	if length > len(lanes) {
		panic("vector: SIMD vector length exceeds supported maximum")
	}
	buf := lanes[:length]
	v.Store(buf)

	var sum float64
	for _, lane := range buf {
		sum += lane
	}
	return sum
}

// Dot returns the inner product of a and b over the shorter of the two slices.
func Dot(a, b []float64) float64 {
	n := min(len(a), len(b))
	length := vectorLength()

	var accumulator simd.Float64s
	i := 0
	for ; i+length <= n; i += length {
		va := simd.LoadFloat64s(a[i:])
		vb := simd.LoadFloat64s(b[i:])
		accumulator = va.MulAdd(vb, accumulator)
	}

	sum := horizontalSum(accumulator, length)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// Add stores a+b into dst over the shorter of the three slices.
func Add(dst, a, b []float64) {
	n := min(len(dst), min(len(a), len(b)))
	length := vectorLength()

	i := 0
	for ; i+length <= n; i += length {
		va := simd.LoadFloat64s(a[i:])
		vb := simd.LoadFloat64s(b[i:])
		va.Add(vb).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

// Sub stores a-b into dst over the shorter of the three slices.
func Sub(dst, a, b []float64) {
	n := min(len(dst), min(len(a), len(b)))
	length := vectorLength()

	i := 0
	for ; i+length <= n; i += length {
		va := simd.LoadFloat64s(a[i:])
		vb := simd.LoadFloat64s(b[i:])
		va.Sub(vb).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

// Scale stores a*factor into dst over the shorter of dst and a.
func Scale(dst, a []float64, factor float64) {
	n := min(len(dst), len(a))
	length := vectorLength()

	factorVector := simd.BroadcastFloat64s(factor)
	i := 0
	for ; i+length <= n; i += length {
		va := simd.LoadFloat64s(a[i:])
		va.Mul(factorVector).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * factor
	}
}

// MulAdd stores dst+a*b*factor into dst over the shorter of the three slices.
func MulAdd(dst, a, b []float64, factor float64) {
	n := min(len(dst), min(len(a), len(b)))
	length := vectorLength()

	factorVector := simd.BroadcastFloat64s(factor)
	i := 0
	for ; i+length <= n; i += length {
		va := simd.LoadFloat64s(a[i:])
		vb := simd.LoadFloat64s(b[i:])
		vd := simd.LoadFloat64s(dst[i:])
		va.MulAdd(vb.Mul(factorVector), vd).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += a[i] * b[i] * factor
	}
}

// Sum returns the sum of all elements of a.
func Sum(a []float64) float64 {
	n := len(a)
	length := vectorLength()

	var accumulator simd.Float64s
	i := 0
	for ; i+length <= n; i += length {
		accumulator = accumulator.Add(simd.LoadFloat64s(a[i:]))
	}

	sum := horizontalSum(accumulator, length)
	for ; i < n; i++ {
		sum += a[i]
	}
	return sum
}
