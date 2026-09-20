package neat

import "sort"

// connectionKey identifies a connection by its endpoints for innovation
// numbering.
type connectionKey struct {
	In  int
	Out int
}

// InnovationTracker assigns stable innovation numbers to connections and IDs
// to newly created nodes.
//
// Evolution is centralized and single-threaded, so the tracker does not need
// internal locking. It is safe to use across a whole population's lifetime.
type InnovationTracker struct {
	nextInnovation int
	nextNodeID     int
	connections    map[connectionKey]int
}

// NewInnovationTracker creates a tracker whose first node ID leaves room for
// the base topology: inputs, one bias node and outputs.
func NewInnovationTracker(inputCount, outputCount int) *InnovationTracker {
	return &InnovationTracker{
		nextInnovation: 1,
		nextNodeID:     inputCount + 1 + outputCount,
		connections:    make(map[connectionKey]int),
	}
}

// ConnectionInnovation returns the innovation number for the connection from
// in to out, allocating a new one if necessary.
func (t *InnovationTracker) ConnectionInnovation(in, out int) int {
	key := connectionKey{In: in, Out: out}
	if innovation, ok := t.connections[key]; ok {
		return innovation
	}
	innovation := t.nextInnovation
	t.nextInnovation++
	t.connections[key] = innovation
	return innovation
}

// NextNodeID allocates a fresh hidden node ID.
func (t *InnovationTracker) NextNodeID() int {
	id := t.nextNodeID
	t.nextNodeID++
	return id
}

// NextInnovation exposes the next innovation number (useful for diagnostics).
func (t *InnovationTracker) NextInnovation() int {
	return t.nextInnovation
}

// reset clears all state; used when restoring a population snapshot.
func (t *InnovationTracker) reset(nextInnovation, nextNodeID int) {
	t.nextInnovation = nextInnovation
	t.nextNodeID = nextNodeID
	t.connections = make(map[connectionKey]int)
}

// observe records an existing connection so duplicate innovations are not
// handed out after a restore.
func (t *InnovationTracker) observe(in, out, innovation int) {
	t.connections[connectionKey{In: in, Out: out}] = innovation
	if innovation >= t.nextInnovation {
		t.nextInnovation = innovation + 1
	}
	if in >= t.nextNodeID {
		t.nextNodeID = in + 1
	}
	if out >= t.nextNodeID {
		t.nextNodeID = out + 1
	}
}

// innovationState is the serialized form of an InnovationTracker.
type innovationState struct {
	NextInnovation int               `json:"next_innovation"`
	NextNodeID     int               `json:"next_node_id"`
	Connections    []connectionState `json:"connections"`
}

// connectionState is one entry of the innovation mapping.
type connectionState struct {
	In         int `json:"in"`
	Out        int `json:"out"`
	Innovation int `json:"innovation"`
}

// MarshalState captures the tracker for a snapshot.
func (t *InnovationTracker) MarshalState() innovationState {
	state := innovationState{
		NextInnovation: t.nextInnovation,
		NextNodeID:     t.nextNodeID,
		Connections:    make([]connectionState, 0, len(t.connections)),
	}
	for key, innovation := range t.connections {
		state.Connections = append(state.Connections, connectionState{
			In:         key.In,
			Out:        key.Out,
			Innovation: innovation,
		})
	}
	// Canonical order so snapshots are byte-stable across runs.
	sort.Slice(state.Connections, func(i, j int) bool {
		if state.Connections[i].Innovation != state.Connections[j].Innovation {
			return state.Connections[i].Innovation < state.Connections[j].Innovation
		}
		if state.Connections[i].In != state.Connections[j].In {
			return state.Connections[i].In < state.Connections[j].In
		}
		return state.Connections[i].Out < state.Connections[j].Out
	})
	return state
}

// UnmarshalState restores a tracker from a snapshot.
func (t *InnovationTracker) UnmarshalState(state innovationState) {
	t.reset(state.NextInnovation, state.NextNodeID)
	for _, connection := range state.Connections {
		t.observe(connection.In, connection.Out, connection.Innovation)
	}
}
