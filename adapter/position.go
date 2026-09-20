package adapter

import "fmt"

// Position2D vectorizes a two-dimensional coordinate by mapping the configured
// world bounds onto [-1, 1] per axis. The adapter owns the current position;
// the surrounding world reads and writes it explicitly.
type Position2D struct {
	name         string
	minimumX     float64
	maximumX     float64
	minimumY     float64
	maximumY     float64
	x            float64
	y            float64
	placement    Point
	hasPlacement bool
}

// NewPosition2D creates a 2D position adapter for a world of the given
// rectangular bounds.
func NewPosition2D(name string, minimumX, maximumX, minimumY, maximumY float64) *Position2D {
	return &Position2D{
		name:     name,
		minimumX: minimumX,
		maximumX: maximumX,
		minimumY: minimumY,
		maximumY: maximumY,
	}
}

// WithPlacement assigns a substrate placement coordinate.
func (p *Position2D) WithPlacement(point Point) *Position2D {
	p.placement = point
	p.hasPlacement = true
	return p
}

// Name returns the adapter name.
func (p *Position2D) Name() string { return p.name }

// Placement implements Placed.
func (p *Position2D) Placement() (Point, bool) { return p.placement, p.hasPlacement }

// Dimensions is always 2.
func (p *Position2D) Dimensions() int { return 2 }

// SetXY sets the current position.
func (p *Position2D) SetXY(x, y float64) {
	p.x = x
	p.y = y
}

// XY returns the current position.
func (p *Position2D) XY() (float64, float64) { return p.x, p.y }

// Sense encodes the current position.
func (p *Position2D) Sense() []float64 {
	return []float64{
		normalize(p.x, p.minimumX, p.maximumX),
		normalize(p.y, p.minimumY, p.maximumY),
	}
}

// Control decodes a normalized vector into the current position, clamped to
// the configured bounds.
func (p *Position2D) Control(values []float64) error {
	if len(values) < 2 {
		return fmt.Errorf("adapter: position2d %q expects 2 values, got %d", p.name, len(values))
	}
	p.x = denormalize(values[0], p.minimumX, p.maximumX)
	p.y = denormalize(values[1], p.minimumY, p.maximumY)
	return nil
}

// Position3D vectorizes a three-dimensional coordinate by mapping the
// configured world bounds onto [-1, 1] per axis.
type Position3D struct {
	name         string
	minimumX     float64
	maximumX     float64
	minimumY     float64
	maximumY     float64
	minimumZ     float64
	maximumZ     float64
	x            float64
	y            float64
	z            float64
	placement    Point
	hasPlacement bool
}

// NewPosition3D creates a 3D position adapter for a world of the given box
// bounds.
func NewPosition3D(name string, minimumX, maximumX, minimumY, maximumY, minimumZ, maximumZ float64) *Position3D {
	return &Position3D{
		name:     name,
		minimumX: minimumX,
		maximumX: maximumX,
		minimumY: minimumY,
		maximumY: maximumY,
		minimumZ: minimumZ,
		maximumZ: maximumZ,
	}
}

// WithPlacement assigns a substrate placement coordinate.
func (p *Position3D) WithPlacement(point Point) *Position3D {
	p.placement = point
	p.hasPlacement = true
	return p
}

// Name returns the adapter name.
func (p *Position3D) Name() string { return p.name }

// Placement implements Placed.
func (p *Position3D) Placement() (Point, bool) { return p.placement, p.hasPlacement }

// Dimensions is always 3.
func (p *Position3D) Dimensions() int { return 3 }

// SetXYZ sets the current position.
func (p *Position3D) SetXYZ(x, y, z float64) {
	p.x = x
	p.y = y
	p.z = z
}

// XYZ returns the current position.
func (p *Position3D) XYZ() (float64, float64, float64) { return p.x, p.y, p.z }

// Sense encodes the current position.
func (p *Position3D) Sense() []float64 {
	return []float64{
		normalize(p.x, p.minimumX, p.maximumX),
		normalize(p.y, p.minimumY, p.maximumY),
		normalize(p.z, p.minimumZ, p.maximumZ),
	}
}

// Control decodes a normalized vector into the current position, clamped to
// the configured bounds.
func (p *Position3D) Control(values []float64) error {
	if len(values) < 3 {
		return fmt.Errorf("adapter: position3d %q expects 3 values, got %d", p.name, len(values))
	}
	p.x = denormalize(values[0], p.minimumX, p.maximumX)
	p.y = denormalize(values[1], p.minimumY, p.maximumY)
	p.z = denormalize(values[2], p.minimumZ, p.maximumZ)
	return nil
}
