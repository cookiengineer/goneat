package adapter

import (
	"math"
	"testing"
)

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= 1e-9
}

func TestEnumOrdinal(t *testing.T) {
	enum := NewEnum("color", []string{"red", "green", "blue"})
	if enum.Dimensions() != 1 {
		t.Fatalf("dimensions = %d, want 1", enum.Dimensions())
	}

	cases := []struct {
		index int
		want  float64
	}{
		{0, -1}, {1, 0}, {2, 1},
	}
	for _, tc := range cases {
		if err := enum.SetIndex(tc.index); err != nil {
			t.Fatalf("SetIndex(%d): %v", tc.index, err)
		}
		if got := enum.Sense()[0]; !almostEqual(got, tc.want) {
			t.Fatalf("index %d encoded to %v, want %v", tc.index, got, tc.want)
		}
	}

	if err := enum.SetValue("blue"); err != nil || enum.Index() != 2 {
		t.Fatalf("SetValue(blue) -> index %d, err %v", enum.Index(), err)
	}
	if err := enum.SetValue("purple"); err == nil {
		t.Fatalf("SetValue(purple) should fail")
	}

	if err := enum.Control([]float64{0.6}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if got := enum.Value(); got != "blue" {
		t.Fatalf("Control(0.6) -> %q, want blue", got)
	}
}

func TestEnumOneHot(t *testing.T) {
	enum := NewEnum("color", []string{"red", "green", "blue"}).AsOneHot()
	if enum.Dimensions() != 3 {
		t.Fatalf("dimensions = %d, want 3", enum.Dimensions())
	}
	_ = enum.SetIndex(1)
	if got := enum.Sense(); got[0] != -1 || got[1] != 1 || got[2] != -1 {
		t.Fatalf("one-hot sense = %v", got)
	}
	if err := enum.Control([]float64{-0.5, 0.2, 0.9}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if enum.Index() != 2 {
		t.Fatalf("Control -> index %d, want 2", enum.Index())
	}
}

func TestEnumPlacement(t *testing.T) {
	enum := NewEnum("color", []string{"a", "b"}).WithPlacement(Point{X: -1, Y: 0})
	point, ok := enum.Placement()
	if !ok || point.X != -1 {
		t.Fatalf("placement = %v, %v", point, ok)
	}
}

func TestStringSimilarity(t *testing.T) {
	if !almostEqual(Similarity("hello", "hello"), 1) {
		t.Fatalf("identical similarity != 1")
	}
	if Similarity("hello", "world") >= Similarity("hello", "hallo") {
		t.Fatalf("expected hello/hallo more similar than hello/world")
	}
	if Similarity("", "") != 1 {
		t.Fatalf("empty similarity should be 1")
	}
}

func TestStringRoundTrip(t *testing.T) {
	adapter := NewString("word", []StringAnchor{
		{Value: "left", Position: -1},
		{Value: "center", Position: 0},
		{Value: "right", Position: 1},
	})

	adapter.SetValue("left")
	encoded := adapter.Sense()
	if len(encoded) != 1 || encoded[0] > -0.5 {
		t.Fatalf("left encoded to %v, want near -1", encoded)
	}

	if err := adapter.Control([]float64{0.9}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if adapter.Value() != "right" {
		t.Fatalf("Control(0.9) -> %q, want right", adapter.Value())
	}
}

func TestPosition2DRoundTrip(t *testing.T) {
	position := NewPosition2D("ball", -100, 100, -50, 50)
	position.SetXY(0, 0)
	if got := position.Sense(); !almostEqual(got[0], 0) || !almostEqual(got[1], 0) {
		t.Fatalf("center sense = %v", got)
	}
	position.SetXY(-100, 50)
	if got := position.Sense(); !almostEqual(got[0], -1) || !almostEqual(got[1], 1) {
		t.Fatalf("corner sense = %v", got)
	}

	if err := position.Control([]float64{1, -1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if x, y := position.XY(); x != 100 || y != -50 {
		t.Fatalf("Control -> (%v,%v)", x, y)
	}

	// Values outside [-1,1] are clamped.
	if err := position.Control([]float64{5, -5}); err != nil {
		t.Fatalf("Control clamp: %v", err)
	}
	if x, y := position.XY(); x != 100 || y != -50 {
		t.Fatalf("clamped Control -> (%v,%v)", x, y)
	}
}

func TestPosition3DRoundTrip(t *testing.T) {
	position := NewPosition3D("bot", 0, 10, 0, 10, 0, 10)
	position.SetXYZ(5, 5, 5)
	if got := position.Sense(); !almostEqual(got[0], 0) || !almostEqual(got[1], 0) || !almostEqual(got[2], 0) {
		t.Fatalf("center sense = %v", got)
	}
	if err := position.Control([]float64{-1, 1, -1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if x, y, z := position.XYZ(); x != 0 || y != 10 || z != 0 {
		t.Fatalf("Control -> (%v,%v,%v)", x, y, z)
	}
}

func TestScalarClamp(t *testing.T) {
	scalar := NewClampedScalar("speed", 0, 10)
	scalar.SetValue(5)
	if got := scalar.Sense()[0]; !almostEqual(got, 0) {
		t.Fatalf("mid sense = %v", got)
	}
	if err := scalar.Control([]float64{1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if scalar.Value() != 10 {
		t.Fatalf("Control(1) -> %v, want 10", scalar.Value())
	}
}

func TestScalarTanh(t *testing.T) {
	scalar := NewSquashedScalar("unbounded", 2)
	scalar.SetValue(0)
	if got := scalar.Sense()[0]; !almostEqual(got, 0) {
		t.Fatalf("zero sense = %v", got)
	}
	if err := scalar.Control([]float64{0.5}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if !almostEqual(scalar.Value(), math.Atanh(0.5)*2) {
		t.Fatalf("Control(0.5) -> %v", scalar.Value())
	}
}

func TestBool(t *testing.T) {
	boolean := NewBool("jump")
	boolean.SetValue(true)
	if got := boolean.Sense()[0]; got != 1 {
		t.Fatalf("true sense = %v", got)
	}
	boolean.SetValue(false)
	if got := boolean.Sense()[0]; got != -1 {
		t.Fatalf("false sense = %v", got)
	}
	if err := boolean.Control([]float64{-0.1}); err != nil {
		t.Fatalf("Control: %v", err)
	}
	if boolean.Value() {
		t.Fatalf("Control(-0.1) -> true, want false")
	}
}

func TestVectorizerManifestAndFrames(t *testing.T) {
	position := NewPosition2D("paddle", -1, 1, -1, 1).WithPlacement(Point{X: -1, Y: 0})
	enum := NewEnum("state", []string{"idle", "run", "jump"})
	boolean := NewBool("alive")

	vectorizer, err := NewVectorizer(
		[]Sensor{position, enum, boolean},
		[]Control{position},
	)
	if err != nil {
		t.Fatalf("NewVectorizer: %v", err)
	}

	manifest := vectorizer.Manifest()
	if manifest.InputDimensions != 2+1+1 {
		t.Fatalf("input dimensions = %d, want 4", manifest.InputDimensions)
	}
	if manifest.OutputDimensions != 2 {
		t.Fatalf("output dimensions = %d, want 2", manifest.OutputDimensions)
	}
	if err := manifest.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	entry, ok := manifest.SensorEntry("state")
	if !ok || entry.Offset != 2 || entry.Dimensions != 1 {
		t.Fatalf("state entry = %+v, %v", entry, ok)
	}
	if !entry.HasPlacement {
		// enum has no placement; position does.
	}
	paddleEntry, _ := manifest.SensorEntry("paddle")
	if !paddleEntry.HasPlacement || paddleEntry.Placement.X != -1 {
		t.Fatalf("paddle placement = %+v", paddleEntry)
	}

	position.SetXY(-1, 1)
	_ = enum.SetIndex(2)
	boolean.SetValue(true)
	frame := vectorizer.Sense(nil)
	if len(frame) != 4 {
		t.Fatalf("frame length = %d, want 4", len(frame))
	}
	if frame[0] != -1 || frame[1] != 1 || frame[2] != 1 || frame[3] != 1 {
		t.Fatalf("frame = %v", frame)
	}

	if err := vectorizer.ApplyControls([]float64{0.5, -0.5}); err != nil {
		t.Fatalf("ApplyControls: %v", err)
	}
	if x, y := position.XY(); !almostEqual(x, 0.5) || !almostEqual(y, -0.5) {
		t.Fatalf("applied position = (%v,%v)", x, y)
	}
}

func TestVectorizerRejectsDuplicates(t *testing.T) {
	a := NewBool("same")
	b := NewBool("same")
	if _, err := NewVectorizer([]Sensor{a, b}, []Control{a}); err == nil {
		t.Fatalf("expected duplicate name error")
	}
}
