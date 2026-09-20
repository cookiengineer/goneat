// Package main implements a headless Pong environment for goneat.
//
// The world hosts N independent 1-v-1 lanes. Every agent controls the left
// ("good") paddle of its own lane through one normalized control value and
// observes the ball and its own paddle through five normalized sensor values.
// A scripted opponent controls the right paddle. This keeps every agent
// independent, so an entire population can train in parallel inside one world.
package main

import (
	"encoding/json"
	"math"

	"github.com/cookiengineer/goneat/engine/random"
)

const (
	paddleX         = 0.9
	paddleHalfSize  = 0.25
	paddleSpeed     = 2.0
	opponentSpeed   = 0.4
	ballSpeed       = 0.7
	ballHitOffset   = 0.05
	scoreBoundary   = 1.15
	pongSensorCount = 5
)

// Event codes reported by a lane after Step.
const (
	EventNone  = 0.0
	EventHit   = 1.0
	EventMiss  = -1.0
	EventScore = 3.0
)

type pongLane struct {
	ballX, ballY   float64
	ballVX, ballVY float64
	goodY          float64
	evilY          float64
	goodTarget     float64
	event          float64
	score          int
	conceded       int
	hits           int
}

func (l *pongLane) serve(rng *random.Rand) {
	l.ballX = 0
	l.ballY = rng.Float64()*1.2 - 0.6
	l.goodY = 0
	l.evilY = 0
	l.goodTarget = 0

	direction := 1.0
	if rng.Float64() < 0.5 {
		direction = -1
	}
	angle := (rng.Float64()*0.8 - 0.4) // radians-ish spread
	l.ballVX = direction * ballSpeed * math.Cos(angle)
	l.ballVY = ballSpeed * math.Sin(angle)
}

// PongWorld is a simulation.World implementation.
type PongWorld struct {
	agentCount int
	maxTicks   int
	ticks      int
	lanes      []pongLane
	rng        *random.Rand
}

// NewPongWorld creates a world for agentCount lanes.
func NewPongWorld(agentCount, maxTicks int) *PongWorld {
	return &PongWorld{agentCount: agentCount, maxTicks: maxTicks}
}

// SensorDimensions implements simulation.World.
func (w *PongWorld) SensorDimensions() int { return pongSensorCount }

// ControlDimensions implements simulation.World.
func (w *PongWorld) ControlDimensions() int { return 1 }

// Reset implements simulation.World.
func (w *PongWorld) Reset(seed int64) {
	w.ticks = 0
	w.rng = random.New(seed)
	if len(w.lanes) != w.agentCount {
		w.lanes = make([]pongLane, w.agentCount)
	}
	for i := range w.lanes {
		w.lanes[i] = pongLane{}
		w.lanes[i].serve(w.rng)
	}
}

// Sensors implements simulation.World.
func (w *PongWorld) Sensors() [][]float64 {
	frames := make([][]float64, w.agentCount)
	for i := range frames {
		lane := &w.lanes[i]
		frames[i] = []float64{
			clamp(lane.ballX),
			clamp(lane.ballY),
			clamp(lane.ballVX / ballSpeed),
			clamp(lane.ballVY / ballSpeed),
			clamp(lane.goodY),
		}
	}
	return frames
}

// Apply implements simulation.World.
func (w *PongWorld) Apply(controls [][]float64) {
	for i := range w.lanes {
		if i >= len(controls) || len(controls[i]) == 0 {
			continue
		}
		w.lanes[i].goodTarget = clamp(controls[i][0])
	}
}

// Step implements simulation.World.
func (w *PongWorld) Step(dt float64) {
	for i := range w.lanes {
		w.stepLane(&w.lanes[i], dt)
	}
	w.ticks++
}

func (w *PongWorld) stepLane(lane *pongLane, dt float64) {
	lane.event = EventNone

	// Agent paddle moves toward its requested target.
	movePaddle(&lane.goodY, lane.goodTarget, paddleSpeed*dt)
	// Scripted opponent tracks the ball.
	movePaddle(&lane.evilY, lane.ballY, opponentSpeed*dt)

	lane.ballX += lane.ballVX * dt
	lane.ballY += lane.ballVY * dt

	if lane.ballY > 1 {
		lane.ballY = 1
		lane.ballVY = -math.Abs(lane.ballVY)
	} else if lane.ballY < -1 {
		lane.ballY = -1
		lane.ballVY = math.Abs(lane.ballVY)
	}

	// Left paddle (agent).
	if lane.ballVX < 0 && lane.ballX <= -paddleX+ballHitOffset && lane.ballX >= -paddleX-ballHitOffset {
		if math.Abs(lane.ballY-lane.goodY) <= paddleHalfSize {
			lane.ballX = -paddleX + ballHitOffset
			lane.ballVX = math.Abs(lane.ballVX)
			offset := (lane.ballY - lane.goodY) / paddleHalfSize
			lane.ballVY = ballSpeed * clamp(offset)
			lane.event = EventHit
			lane.hits++
		}
	}

	// Right paddle (opponent).
	if lane.ballVX > 0 && lane.ballX >= paddleX-ballHitOffset && lane.ballX <= paddleX+ballHitOffset {
		if math.Abs(lane.ballY-lane.evilY) <= paddleHalfSize {
			lane.ballX = paddleX - ballHitOffset
			lane.ballVX = -math.Abs(lane.ballVX)
			offset := (lane.ballY - lane.evilY) / paddleHalfSize
			lane.ballVY = ballSpeed * clamp(offset)
		}
	}

	if lane.ballX < -scoreBoundary {
		lane.conceded++
		lane.event = EventMiss
		lane.serve(w.rng)
	} else if lane.ballX > scoreBoundary {
		lane.score++
		lane.event = EventScore
		lane.serve(w.rng)
	}
}

// Done implements simulation.World.
func (w *PongWorld) Done() []bool {
	done := make([]bool, w.agentCount)
	for i := range done {
		done[i] = w.ticks >= w.maxTicks
	}
	return done
}

// Event returns the last event for a lane.
func (w *PongWorld) Event(agentIndex int) float64 {
	if agentIndex < 0 || agentIndex >= len(w.lanes) {
		return EventNone
	}
	return w.lanes[agentIndex].event
}

// BallY returns the normalized ball height in a lane, the natural target for
// the paddle.
func (w *PongWorld) BallY(agentIndex int) float64 {
	if agentIndex < 0 || agentIndex >= len(w.lanes) {
		return 0
	}
	return clamp(w.lanes[agentIndex].ballY)
}

// Hits returns how many times an agent has returned the ball in this episode.
func (w *PongWorld) Hits(agentIndex int) int {
	if agentIndex < 0 || agentIndex >= len(w.lanes) {
		return 0
	}
	return w.lanes[agentIndex].hits
}

// Scores returns the total good/opponent points across all lanes.
func (w *PongWorld) Scores() (good, evil int) {
	for i := range w.lanes {
		good += w.lanes[i].score
		evil += w.lanes[i].conceded
	}
	return good, evil
}

// Tick returns the current tick.
func (w *PongWorld) Tick() int { return w.ticks }

type pongLaneState struct {
	BallX    float64 `json:"ball_x"`
	BallY    float64 `json:"ball_y"`
	BallVX   float64 `json:"ball_vx"`
	BallVY   float64 `json:"ball_vy"`
	GoodY    float64 `json:"good_y"`
	EvilY    float64 `json:"evil_y"`
	Event    float64 `json:"event"`
	Score    int     `json:"score"`
	Conceded int     `json:"conceded"`
	Hits     int     `json:"hits"`
}

type pongWorldState struct {
	AgentCount int             `json:"agent_count"`
	MaxTicks   int             `json:"max_ticks"`
	Ticks      int             `json:"ticks"`
	Lanes      []pongLaneState `json:"lanes"`
	RNG        *random.Rand    `json:"rng"`
}

// Snapshot implements simulation.World.
func (w *PongWorld) Snapshot() ([]byte, error) {
	lanes := make([]pongLaneState, len(w.lanes))
	for i, lane := range w.lanes {
		lanes[i] = pongLaneState{
			BallX: lane.ballX, BallY: lane.ballY,
			BallVX: lane.ballVX, BallVY: lane.ballVY,
			GoodY: lane.goodY, EvilY: lane.evilY,
			Event: lane.event, Score: lane.score,
			Conceded: lane.conceded, Hits: lane.hits,
		}
	}
	return json.Marshal(pongWorldState{
		AgentCount: w.agentCount,
		MaxTicks:   w.maxTicks,
		Ticks:      w.ticks,
		Lanes:      lanes,
		RNG:        w.rng,
	})
}

// Restore implements simulation.World.
func (w *PongWorld) Restore(data []byte) error {
	var state pongWorldState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	w.agentCount = state.AgentCount
	w.maxTicks = state.MaxTicks
	w.ticks = state.Ticks
	w.lanes = make([]pongLane, len(state.Lanes))
	for i, lane := range state.Lanes {
		w.lanes[i] = pongLane{
			ballX: lane.BallX, ballY: lane.BallY,
			ballVX: lane.BallVX, ballVY: lane.BallVY,
			goodY: lane.GoodY, evilY: lane.EvilY,
			event: lane.Event, score: lane.Score,
			conceded: lane.Conceded, hits: lane.Hits,
		}
	}
	if state.RNG != nil {
		w.rng = state.RNG
	} else {
		w.rng = random.New(1)
	}
	return nil
}

func movePaddle(position *float64, target, maxStep float64) {
	delta := target - *position
	if delta > maxStep {
		delta = maxStep
	} else if delta < -maxStep {
		delta = -maxStep
	}
	*position += delta
	if *position > 1-paddleHalfSize {
		*position = 1 - paddleHalfSize
	} else if *position < -1+paddleHalfSize {
		*position = -1 + paddleHalfSize
	}
}

func clamp(value float64) float64 {
	if value > 1 {
		return 1
	}
	if value < -1 {
		return -1
	}
	return value
}
