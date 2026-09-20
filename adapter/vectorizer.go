package adapter

import "fmt"

// EntryKind distinguishes sensor and control slots in a Manifest.
type EntryKind string

const (
	KindSensor  EntryKind = "sensor"
	KindControl EntryKind = "control"
)

// Entry describes one adapter within a flattened frame.
type Entry struct {
	Name         string    `json:"name"`
	Kind         EntryKind `json:"kind"`
	Offset       int       `json:"offset"`
	Dimensions   int       `json:"dimensions"`
	Placement    Point     `json:"placement"`
	HasPlacement bool      `json:"has_placement"`
}

// Manifest is the flat layout of a Vectorizer. It is serialized alongside a
// brain so a decoded brain reconnects to the same slots.
type Manifest struct {
	Sensors          []Entry `json:"sensors"`
	Controls         []Entry `json:"controls"`
	InputDimensions  int     `json:"input_dimensions"`
	OutputDimensions int     `json:"output_dimensions"`
}

// SensorEntry finds a sensor manifest entry by name.
func (m Manifest) SensorEntry(name string) (Entry, bool) {
	for _, entry := range m.Sensors {
		if entry.Name == name {
			return entry, true
		}
	}
	return Entry{}, false
}

// ControlEntry finds a control manifest entry by name.
func (m Manifest) ControlEntry(name string) (Entry, bool) {
	for _, entry := range m.Controls {
		if entry.Name == name {
			return entry, true
		}
	}
	return Entry{}, false
}

// Validate checks that the manifest is internally consistent.
func (m Manifest) Validate() error {
	if len(m.Sensors) == 0 {
		return fmt.Errorf("adapter: manifest has no sensors")
	}
	if len(m.Controls) == 0 {
		return fmt.Errorf("adapter: manifest has no controls")
	}
	if m.InputDimensions <= 0 || m.OutputDimensions <= 0 {
		return fmt.Errorf("adapter: manifest dimensions must be positive")
	}
	return nil
}

// Vectorizer aggregates sensors and controls and flattens them into frames.
type Vectorizer struct {
	sensors  []Sensor
	controls []Control
	manifest Manifest
}

// NewVectorizer builds a vectorizer and computes its manifest. It returns an
// error if any adapter reports non-positive dimensions or a duplicate name.
func NewVectorizer(sensors []Sensor, controls []Control) (*Vectorizer, error) {
	vectorizer := &Vectorizer{
		sensors:  append([]Sensor(nil), sensors...),
		controls: append([]Control(nil), controls...),
	}

	seenSensors := map[string]bool{}
	seenControls := map[string]bool{}

	offset := 0
	for _, sensor := range sensors {
		if sensor == nil {
			return nil, fmt.Errorf("adapter: nil sensor")
		}
		if sensor.Dimensions() <= 0 {
			return nil, fmt.Errorf("adapter: sensor %q has non-positive dimensions", sensor.Name())
		}
		if seenSensors[sensor.Name()] {
			return nil, fmt.Errorf("adapter: duplicate sensor name %q", sensor.Name())
		}
		seenSensors[sensor.Name()] = true

		entry := Entry{
			Name:       sensor.Name(),
			Kind:       KindSensor,
			Offset:     offset,
			Dimensions: sensor.Dimensions(),
		}
		if placed, ok := sensor.(Placed); ok {
			entry.Placement, entry.HasPlacement = placed.Placement()
		}
		vectorizer.manifest.Sensors = append(vectorizer.manifest.Sensors, entry)
		offset += entry.Dimensions
	}
	vectorizer.manifest.InputDimensions = offset

	offset = 0
	for _, control := range controls {
		if control == nil {
			return nil, fmt.Errorf("adapter: nil control")
		}
		if control.Dimensions() <= 0 {
			return nil, fmt.Errorf("adapter: control %q has non-positive dimensions", control.Name())
		}
		if seenControls[control.Name()] {
			return nil, fmt.Errorf("adapter: duplicate control name %q", control.Name())
		}
		seenControls[control.Name()] = true

		entry := Entry{
			Name:       control.Name(),
			Kind:       KindControl,
			Offset:     offset,
			Dimensions: control.Dimensions(),
		}
		if placed, ok := control.(Placed); ok {
			entry.Placement, entry.HasPlacement = placed.Placement()
		}
		vectorizer.manifest.Controls = append(vectorizer.manifest.Controls, entry)
		offset += entry.Dimensions
	}
	vectorizer.manifest.OutputDimensions = offset

	if err := vectorizer.manifest.Validate(); err != nil {
		return nil, err
	}
	return vectorizer, nil
}

// Manifest returns the computed layout.
func (v *Vectorizer) Manifest() Manifest {
	return v.manifest
}

// Sensors returns the registered sensors.
func (v *Vectorizer) Sensors() []Sensor {
	return v.sensors
}

// Controls returns the registered controls.
func (v *Vectorizer) Controls() []Control {
	return v.controls
}

// InputDimensions returns the size of a sensor frame.
func (v *Vectorizer) InputDimensions() int {
	return v.manifest.InputDimensions
}

// OutputDimensions returns the size of a control frame.
func (v *Vectorizer) OutputDimensions() int {
	return v.manifest.OutputDimensions
}

// Sense concatenates all sensor readings into frame. It grows frame if it is
// too small and returns the populated slice. Reusing a caller-owned buffer
// avoids allocations in the hot path.
func (v *Vectorizer) Sense(frame []float64) []float64 {
	required := v.manifest.InputDimensions
	if cap(frame) < required {
		frame = make([]float64, required)
	} else {
		frame = frame[:required]
	}

	offset := 0
	for _, sensor := range v.sensors {
		values := sensor.Sense()
		copy(frame[offset:], values)
		offset += sensor.Dimensions()
	}
	return frame
}

// ApplyControls slices a control frame and forwards each part to its control.
func (v *Vectorizer) ApplyControls(frame []float64) error {
	if len(frame) < v.manifest.OutputDimensions {
		return fmt.Errorf("adapter: control frame too short: %d < %d", len(frame), v.manifest.OutputDimensions)
	}
	for _, control := range v.controls {
		entry, ok := v.manifest.ControlEntry(control.Name())
		if !ok {
			return fmt.Errorf("adapter: missing manifest entry for control %q", control.Name())
		}
		if err := control.Control(frame[entry.Offset : entry.Offset+entry.Dimensions]); err != nil {
			return fmt.Errorf("adapter: control %q: %w", control.Name(), err)
		}
	}
	return nil
}
