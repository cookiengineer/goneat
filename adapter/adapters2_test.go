package adapter

import (
	"math"
	"testing"
	"time"
)

func TestIntAdapterRoundTrip(t *testing.T) {
	adapter := NewInt("signed", -10, 10)
	cases := []struct {
		value    int64
		expected float64
	}{{-10, -1}, {0, 0}, {10, 1}}
	for _, tc := range cases {
		adapter.SetValue(tc.value)
		if got := adapter.Sense()[0]; !almostEqual(got, tc.expected) {
			t.Fatalf("IntAdapter.Sense(%d) = %v, want %v", tc.value, got, tc.expected)
		}
	}

	if err := adapter.Control([]float64{1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != 10 {
		t.Fatalf("Control(1) -> %d, want 10", adapter.Value())
	}
	if err := adapter.Control([]float64{-1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != -10 {
		t.Fatalf("Control(-1) -> %d, want -10", adapter.Value())
	}

	adapter.SetValue(1000)
	if adapter.Value() != 10 {
		t.Fatalf("SetValue clamp -> %d, want 10", adapter.Value())
	}
	if err := adapter.Control(nil); err == nil {
		t.Fatalf("Control with no values should fail")
	}
}

func TestUintAdapterRoundTrip(t *testing.T) {
	adapter := NewUint("unsigned", 0, 100)
	adapter.SetValue(50)
	if got := adapter.Sense()[0]; !almostEqual(got, 0) {
		t.Fatalf("mid sense = %v, want 0", got)
	}
	if err := adapter.Control([]float64{1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != 100 {
		t.Fatalf("Control(1) -> %d, want 100", adapter.Value())
	}
	if err := adapter.Control([]float64{-1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != 0 {
		t.Fatalf("Control(-1) -> %d, want 0", adapter.Value())
	}
}

func TestFloat32Adapter(t *testing.T) {
	clamped := NewClampedFloat32("clamped", 0, 10)
	clamped.SetValue(5)
	if got := clamped.Sense()[0]; !almostEqual(got, 0) {
		t.Fatalf("clamped mid sense = %v", got)
	}
	if err := clamped.Control([]float64{1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if clamped.Value() != 10 {
		t.Fatalf("Control(1) -> %v, want 10", clamped.Value())
	}

	squashed := NewSquashedFloat32("squashed", 2)
	if err := squashed.Control([]float64{0.5}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	want := math.Atanh(0.5) * 2
	if math.Abs(float64(squashed.Value())-want) > 1e-4 {
		t.Fatalf("squashed Control(0.5) = %v, want %v", squashed.Value(), want)
	}
}

func TestByteAdapter(t *testing.T) {
	adapter := NewByte("octet")
	if err := adapter.Control([]float64{-1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != 0 {
		t.Fatalf("Control(-1) -> %d, want 0", adapter.Value())
	}
	if err := adapter.Control([]float64{1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != 255 {
		t.Fatalf("Control(1) -> %d, want 255", adapter.Value())
	}
	adapter.SetValue(128)
	if got := adapter.Sense()[0]; !almostEqual(got, normalize(128, 0, 255)) {
		t.Fatalf("sense = %v", got)
	}
}

func TestRuneAdapter(t *testing.T) {
	adapter := NewRuneRange("ascii", 32, 126)
	adapter.SetValue('A')
	encoded := adapter.Sense()[0]
	if err := adapter.Control([]float64{encoded}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != 'A' {
		t.Fatalf("round trip -> %q, want A", adapter.Value())
	}

	adapter.SetValue('x' + 1000)
	if adapter.Value() != 126 {
		t.Fatalf("SetValue clamp -> %d, want 126", adapter.Value())
	}
}

func TestComplexAdapter(t *testing.T) {
	adapter := NewComplex("signal", 2, 4)
	adapter.SetValue(complex(0.5, -0.8))
	sense := adapter.Sense()
	if !almostEqual(sense[0], math.Tanh(0.25)) || !almostEqual(sense[1], math.Tanh(-0.2)) {
		t.Fatalf("sense = %v", sense)
	}
	if err := adapter.Control(sense); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if math.Abs(real(adapter.Value())-0.5) > 1e-9 || math.Abs(imag(adapter.Value())+0.8) > 1e-9 {
		t.Fatalf("round trip = %v", adapter.Value())
	}
	if adapter.Dimensions() != 2 {
		t.Fatalf("dimensions = %d, want 2", adapter.Dimensions())
	}
}

func TestDurationAdapter(t *testing.T) {
	adapter := NewDuration("timeout", 0, time.Second)
	adapter.SetValue(500 * time.Millisecond)
	if got := adapter.Sense()[0]; !almostEqual(got, 0) {
		t.Fatalf("mid sense = %v, want 0", got)
	}
	if err := adapter.Control([]float64{1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != time.Second {
		t.Fatalf("Control(1) -> %v, want 1s", adapter.Value())
	}
	if err := adapter.Control([]float64{-1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != 0 {
		t.Fatalf("Control(-1) -> %v, want 0", adapter.Value())
	}
}

func TestSliceAdapter(t *testing.T) {
	adapter := NewSlice("observation", 3, 2)
	if adapter.Dimensions() != 3 {
		t.Fatalf("dimensions = %d", adapter.Dimensions())
	}
	if err := adapter.SetValues([]float64{0, 2, 4}); err != nil {
		t.Fatalf("SetValues: %v", err)
	}
	sense := adapter.Sense()
	if !almostEqual(sense[0], 0) || !almostEqual(sense[1], 1) || !almostEqual(sense[2], 1) {
		t.Fatalf("sense = %v, want [0 1 1]", sense)
	}
	if err := adapter.Control([]float64{0, 0.5, 1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	values := adapter.Values()
	if !almostEqual(values[0], 0) || !almostEqual(values[1], 1) || !almostEqual(values[2], 2) {
		t.Fatalf("control values = %v, want [0 1 2]", values)
	}
	if err := adapter.SetValues([]float64{1}); err == nil {
		t.Fatalf("SetValues with wrong length should fail")
	}
}

func TestMapAdapter(t *testing.T) {
	adapter := NewMap("stats", []string{"a", "b", "c"}, -1, 1)
	adapter.SetValues(map[string]float64{"a": 0, "b": 1, "c": -1})
	sense := adapter.Sense()
	if !almostEqual(sense[0], 0) || !almostEqual(sense[1], 1) || !almostEqual(sense[2], -1) {
		t.Fatalf("sense = %v", sense)
	}
	if err := adapter.Control([]float64{0, 0.5, -0.5}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	values := adapter.Values()
	if !almostEqual(values["a"], 0) || !almostEqual(values["b"], 0.5) || !almostEqual(values["c"], -0.5) {
		t.Fatalf("control values = %v", values)
	}

	// Unknown keys are ignored and missing keys read as zero.
	adapter.SetValues(map[string]float64{"a": 1, "zzz": 99})
	if got := adapter.Sense(); !almostEqual(got[0], 1) || !almostEqual(got[1], 0) {
		t.Fatalf("sense with unknown key = %v", got)
	}
}

func TestNewAdaptersInVectorizer(t *testing.T) {
	integer := NewInt("level", 0, 10)
	observation := NewSlice("observation", 2, 1)
	command := NewMap("command", []string{"x", "y"}, -1, 1)

	vectorizer, err := NewVectorizer(
		[]Sensor{integer, observation},
		[]Control{command},
	)
	if err != nil {
		t.Fatalf("NewVectorizer: %v", err)
	}
	if got := vectorizer.InputDimensions(); got != 3 {
		t.Fatalf("input dimensions = %d, want 3", got)
	}
	if got := vectorizer.OutputDimensions(); got != 2 {
		t.Fatalf("output dimensions = %d, want 2", got)
	}

	integer.SetValue(10)
	if err := observation.SetValues([]float64{0.5, -0.5}); err != nil {
		t.Fatalf("SetValues: %v", err)
	}
	frame := vectorizer.Sense(nil)
	if !almostEqual(frame[0], 1) || !almostEqual(frame[1], 0.5) || !almostEqual(frame[2], -0.5) {
		t.Fatalf("frame = %v", frame)
	}

	if err := vectorizer.ApplyControls([]float64{1, -1}); err != nil {
		t.Fatalf("ApplyControls: %v", err)
	}
	values := command.Values()
	if !almostEqual(values["x"], 1) || !almostEqual(values["y"], -1) {
		t.Fatalf("applied command = %v", values)
	}
}
