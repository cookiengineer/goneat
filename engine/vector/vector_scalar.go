//go:build !goexperiment.simd

package vector

// scale_unroll is the scalar unroll factor.
const scale_unroll = 4

// Dot returns the inner product of a and b over the shorter of the two slices.
func Dot(a, b []float64) float64 {
	n := min(len(a), len(b))

	var sum0, sum1, sum2, sum3 float64
	i := 0
	for ; i+scale_unroll <= n; i += scale_unroll {
		sum0 += a[i+0] * b[i+0]
		sum1 += a[i+1] * b[i+1]
		sum2 += a[i+2] * b[i+2]
		sum3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		sum0 += a[i] * b[i]
	}

	return (sum0 + sum1) + (sum2 + sum3)
}

// Add stores a+b into dst over the shorter of the three slices.
func Add(dst, a, b []float64) {
	n := min(len(dst), min(len(a), len(b)))

	i := 0
	for ; i+scale_unroll <= n; i += scale_unroll {
		dst[i+0] = a[i+0] + b[i+0]
		dst[i+1] = a[i+1] + b[i+1]
		dst[i+2] = a[i+2] + b[i+2]
		dst[i+3] = a[i+3] + b[i+3]
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

// Sub stores a-b into dst over the shorter of the three slices.
func Sub(dst, a, b []float64) {
	n := min(len(dst), min(len(a), len(b)))

	i := 0
	for ; i+scale_unroll <= n; i += scale_unroll {
		dst[i+0] = a[i+0] - b[i+0]
		dst[i+1] = a[i+1] - b[i+1]
		dst[i+2] = a[i+2] - b[i+2]
		dst[i+3] = a[i+3] - b[i+3]
	}
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

// Scale stores a*factor into dst over the shorter of dst and a.
func Scale(dst, a []float64, factor float64) {
	n := min(len(dst), len(a))

	i := 0
	for ; i+scale_unroll <= n; i += scale_unroll {
		dst[i+0] = a[i+0] * factor
		dst[i+1] = a[i+1] * factor
		dst[i+2] = a[i+2] * factor
		dst[i+3] = a[i+3] * factor
	}
	for ; i < n; i++ {
		dst[i] = a[i] * factor
	}
}

// MulAdd stores dst+a*b*factor into dst over the shorter of the three slices.
func MulAdd(dst, a, b []float64, factor float64) {
	n := min(len(dst), min(len(a), len(b)))

	i := 0
	for ; i+scale_unroll <= n; i += scale_unroll {
		dst[i+0] += a[i+0] * b[i+0] * factor
		dst[i+1] += a[i+1] * b[i+1] * factor
		dst[i+2] += a[i+2] * b[i+2] * factor
		dst[i+3] += a[i+3] * b[i+3] * factor
	}
	for ; i < n; i++ {
		dst[i] += a[i] * b[i] * factor
	}
}

// Sum returns the sum of all elements of a.
func Sum(a []float64) float64 {
	var sum0, sum1, sum2, sum3 float64
	i := 0
	n := len(a)
	for ; i+scale_unroll <= n; i += scale_unroll {
		sum0 += a[i+0]
		sum1 += a[i+1]
		sum2 += a[i+2]
		sum3 += a[i+3]
	}
	for ; i < n; i++ {
		sum0 += a[i]
	}
	return (sum0 + sum1) + (sum2 + sum3)
}
