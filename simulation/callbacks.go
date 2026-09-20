package simulation

import "github.com/cookiengineer/goneat/agent"

// StepContext gives a Reward/Punish callback access to the current tick, the
// raw world (so the user can measure arbitrary, non-vectorized conditions) and
// the normalized frames that were exchanged.
type StepContext struct {
	Tick     int
	World    World
	Sensors  [][]float64
	Controls [][]float64
	AgentIDs []string
}

// AgentIndex returns the index of an agent ID, or -1.
func (s *StepContext) AgentIndex(id string) int {
	for index, candidate := range s.AgentIDs {
		if candidate == id {
			return index
		}
	}
	return -1
}

// RewardFunc is a user-supplied callback invoked once per agent per tick. It
// returns a fitness delta and, optionally, a supervised training sample in the
// normalized world. Returning a zero delta and nil training is a no-op.
type RewardFunc func(step *StepContext, agent *agent.Agent) (delta float64, training *agent.Training)

// PunishFunc mirrors RewardFunc for negative feedback.
type PunishFunc func(step *StepContext, agent *agent.Agent) (delta float64, training *agent.Training)
