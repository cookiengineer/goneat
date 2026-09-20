// Package random provides a small deterministic pseudo-random generator whose
// full state can be serialized. Using our own generator (instead of
// math/rand) is what makes training snapshots resumable bit-for-bit: the
// generator state travels with the snapshot.
//
// The generator is PCG-XSH-RR (32-bit output, 64-bit state) with a Box-Muller
// normal sampler. It is not cryptographically secure and is not intended to be.
package random

import (
	"encoding/json"
	"errors"
	"math"
	"math/bits"
)

const (
	multiplier = 6364136223846793005
	// defaultIncrement is the default PCG stream selector (must be odd).
	defaultIncrement = 1442695040888963407
)

// Rand is a deterministic, serializable random source.
type Rand struct {
	state     uint64
	increment uint64
	spare     float64
	hasSpare  bool
}

// New creates a generator seeded with the given value.
func New(seed int64) *Rand {
	generator := &Rand{increment: defaultIncrement}
	generator.Seed(seed)
	return generator
}

// Seed resets the generator state from a seed.
func (r *Rand) Seed(seed int64) {
	r.state = 0
	r.increment = (r.increment << 1) | 1
	r.next32()
	r.state += uint64(seed)
	r.next32()
	r.hasSpare = false
	r.spare = 0
}

func (r *Rand) next32() uint32 {
	r.state = r.state*multiplier + r.increment
	xorshifted := uint32(((r.state >> 18) ^ r.state) >> 27)
	rotation := uint32(r.state >> 59)
	return bits.RotateLeft32(xorshifted, -int(rotation))
}

// Uint64 returns the next 64-bit value.
func (r *Rand) Uint64() uint64 {
	high := uint64(r.next32())
	low := uint64(r.next32())
	return high<<32 | low
}

// Int63 returns a non-negative 63-bit integer.
func (r *Rand) Int63() int64 {
	return int64(r.Uint64() >> 1)
}

// Intn returns a value in [0, n).
func (r *Rand) Intn(n int) int {
	if n <= 0 {
		panic("random: Intn requires n > 0")
	}
	return int(r.Uint64() % uint64(n))
}

// Float64 returns a value in [0, 1).
func (r *Rand) Float64() float64 {
	return float64(r.Uint64()>>11) / float64(uint64(1)<<53)
}

// NormFloat64 returns a standard normal sample using Box-Muller, caching the
// second value so the sequence stays deterministic and cheap.
func (r *Rand) NormFloat64() float64 {
	if r.hasSpare {
		r.hasSpare = false
		return r.spare
	}

	u1 := r.Float64()
	u2 := r.Float64()
	if u1 < 1e-300 {
		u1 = 1e-300
	}
	magnitude := math.Sqrt(-2 * math.Log(u1))
	r.spare = magnitude * math.Sin(2*math.Pi*u2)
	r.hasSpare = true
	return magnitude * math.Cos(2*math.Pi*u2)
}

// randState is the serialized form of a Rand.
type randState struct {
	State     uint64  `json:"state"`
	Increment uint64  `json:"increment"`
	Spare     float64 `json:"spare"`
	HasSpare  bool    `json:"has_spare"`
}

// MarshalJSON implements json.Marshaler.
func (r *Rand) MarshalJSON() ([]byte, error) {
	return json.Marshal(randState{
		State:     r.state,
		Increment: r.increment,
		Spare:     r.spare,
		HasSpare:  r.hasSpare,
	})
}

// UnmarshalJSON implements json.Unmarshaler.
func (r *Rand) UnmarshalJSON(data []byte) error {
	var state randState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	if state.Increment == 0 {
		return errors.New("random: serialized increment must be non-zero")
	}
	r.state = state.State
	r.increment = state.Increment
	r.spare = state.Spare
	r.hasSpare = state.HasSpare
	return nil
}

// ResetStream reseeds the generator with a fresh stream (used when cloning a
// population with a distinct seed).
func (r *Rand) ResetStream(seed int64) {
	r.increment = defaultIncrement
	r.Seed(seed)
}
