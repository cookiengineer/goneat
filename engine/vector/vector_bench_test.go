package vector

import (
	"math/rand"
	"testing"
)

const benchmarkSize = 1024

func benchmarkSlices() (dst, a, b []float64) {
	rng := rand.New(rand.NewSource(42))
	dst = randomSlice(rng, benchmarkSize)
	a = randomSlice(rng, benchmarkSize)
	b = randomSlice(rng, benchmarkSize)
	return dst, a, b
}

func BenchmarkDot(b *testing.B) {
	_, a, bb := benchmarkSlices()
	b.ReportAllocs()
	b.ResetTimer()
	var sink float64
	for i := 0; i < b.N; i++ {
		sink += Dot(a, bb)
	}
	_ = sink
}

func BenchmarkAdd(b *testing.B) {
	dst, a, bb := benchmarkSlices()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Add(dst, a, bb)
	}
}

func BenchmarkSub(b *testing.B) {
	dst, a, bb := benchmarkSlices()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sub(dst, a, bb)
	}
}

func BenchmarkScale(b *testing.B) {
	dst, a, _ := benchmarkSlices()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Scale(dst, a, 0.5)
	}
}

func BenchmarkMulAdd(b *testing.B) {
	dst, a, bb := benchmarkSlices()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MulAdd(dst, a, bb, 0.5)
	}
}

func BenchmarkSum(b *testing.B) {
	_, a, _ := benchmarkSlices()
	b.ReportAllocs()
	b.ResetTimer()
	var sink float64
	for i := 0; i < b.N; i++ {
		sink += Sum(a)
	}
	_ = sink
}
