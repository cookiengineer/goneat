# goneat Implementation Guide

This guide shows how to build a training environment with `goneat`. It covers
the common tasks in order: modeling your program as a `World`, translating with
adapters, wiring a `Simulation`, adding gradient training, phased searching,
HyperNEAT, and snapshots.

Every example uses the standard library only. The runnable versions live in
`examples/`.

---

## 1. Add the module

```sh
go get github.com/cookiengineer/goneat
```

The library targets Go 1.27. All packages are standard-library-only. SIMD math
is available when the binary is built with `GOEXPERIMENT=simd`.

---

## 2. Model your program as a `World`

A `World` is the only thing the simulation knows about your program. It exposes
normalized sensor frames and accepts normalized control frames.

```go
type World interface {
    Reset(seed int64)
    SensorDimensions() int
    ControlDimensions() int
    Sensors() [][]float64
    Apply(controls [][]float64)
    Step(dt float64)
    Done() []bool
    Snapshot() ([]byte, error)
    Restore(data []byte) error
}
```

Rules of thumb:

- `Sensors` returns one frame per agent, each `SensorDimensions` long, values
  in `[-1, 1]`.
- `Apply` receives one frame per agent, each `ControlDimensions` long.
- `Done` reports per-agent episode completion. Returning all-`true` ends the
  episode even if the tick budget is not exhausted.
- `Snapshot`/`Restore` only need to round-trip the state you care about; JSON is
  the recommended format. Returning `[]byte("null")` is acceptable for a
  stateless world.
- Keep a seeded `*random.Rand` inside the world for reproducible variation.

```go
type MyWorld struct {
    lanes  []Lane
    ticks  int
    maxTicks int
    rng    *random.Rand
}

func (w *MyWorld) SensorDimensions() int  { return 4 }
func (w *MyWorld) ControlDimensions() int { return 2 }

func (w *MyWorld) Sensors() [][]float64 {
    frames := make([][]float64, len(w.lanes))
    for i, lane := range w.lanes {
        frames[i] = []float64{lane.A, lane.B, lane.C, lane.D} // already [-1,1]
    }
    return frames
}
```

---

## 3. Translate with adapters

Adapters do the `[-1, 1]` conversion so your world can keep natural units
(pixels, degrees, strings, enums).

```go
type Sensor interface {
    Name() string
    Dimensions() int
    Sense() []float64
}

type Control interface {
    Name() string
    Dimensions() int
    Control(values []float64) error
}
```

```go
paddle := adapter.NewPosition2D("paddle", -100, 100, -50, 50)
ball   := adapter.NewPosition2D("ball", -100, 100, -50, 50)
state  := adapter.NewEnum("state", []string{"idle", "run", "jump"})

vectorizer, err := adapter.NewVectorizer(
    []adapter.Sensor{ball, paddle, state},
    []adapter.Control{paddle},
)
if err != nil {
    log.Fatal(err)
}

// Each tick: copy world state into adapters, then sense.
ball.SetXY(ballX, ballY)
paddle.SetXY(0, paddleY)
frame := vectorizer.Sense(nil)            // flat []float64, length InputDimensions

// After the brain produced a control frame:
if err := vectorizer.ApplyControls(outputs); err != nil {
    log.Fatal(err)
}
x, y := paddle.XY()                       // applied back in world units
```

`Vectorizer.Manifest()` describes each slot (name, offset, dimensions). Optional
`adapter.Placed` adapters also expose a coordinate in the unit hypercube for
HyperNEAT.

Available adapters:

| Adapter | Constructor | Data type |
|---|---|---|
| Enum | `NewEnum` | ordered categories (ordinal or one-hot) |
| String | `NewString` | string (similarity to anchor strings) |
| Bool | `NewBool` | bool |
| Scalar | `NewClampedScalar` / `NewSquashedScalar` | float64 |
| Float32 | `NewClampedFloat32` / `NewSquashedFloat32` | float32 |
| Int | `NewInt` | int64 (bounded) |
| Uint | `NewUint` | uint64 (bounded) |
| Byte | `NewByte` | byte (0..255) |
| Rune | `NewRune` / `NewRuneRange` | rune |
| Complex | `NewComplex` / `NewComplexUniform` | complex128 (2 dims) |
| Duration | `NewDuration` | time.Duration |
| Position | `NewPosition2D` / `NewPosition3D` | world coordinates |
| Slice | `NewSlice` / `NewSliceWithScales` | []float64 |
| Map | `NewMap` | map[string]float64 |

```go
level := adapter.NewInt("level", 0, 10)
pixels := adapter.NewSlice("pixels", 16, 1.0)      // 16 already-normalized values
stats := adapter.NewMap("stats", []string{"hp", "mp"}, 0, 100)
timing := adapter.NewDuration("cooldown", 0, 5*time.Second)
```

---

## 4. Build a population

```go
options := neat.DefaultOptions(4 /* inputs */, 2 /* outputs */)
options.Seed = 1
options.PopulationSize = 200
options.AddLinkRate = 0.2
options.AddNodeRate = 0.1
options.SurvivalRate = 0.2
options.Elitism = 1
options.StagnationLimit = 0

population, err := neat.NewPopulation(options)
```

`neat.Options` controls topology, mutation rates, the activation pool
(`ActivationOptions`), output activation, compatibility coefficients, elitism,
stagnation and phased searching. `DefaultOptions` supplies sensible values.

To evolve with a plain fitness function:

```go
fitness := neat.EvaluatorFunc(func(g *neat.Genome) (float64, error) {
    network, err := neat.NewNetwork(g)
    if err != nil {
        return 0, err
    }
    outputs := make([]float64, network.OutputCount())
    score := 0.0
    for _, sample := range samples {
        if err := network.Activate(sample.Inputs, outputs); err != nil {
            return 0, err
        }
        score += accuracy(outputs, sample.Targets)
    }
    return score / float64(len(samples)), nil
})

stats, err := population.Epoch(fitness, observer)
```

`population.BestGenome()` returns the best genome seen so far.

---

## 5. Wire a `Simulation`

A `Simulation` runs a population of agents inside one world per generation.

```go
simulation, err := simulation.NewSimulation(world, population, simulation.Options{
    Ticks:  600,
    DT:     1.0 / 60.0,
    Seed:   7,
    Runner: simulation.ChannelRunner{}, // goroutine + lock-step barrier
    Reward: myReward,                    // user callback (may be nil)
    Punish: myPunish,                    // user callback (may be nil)
})
if err != nil {
    log.Fatal(err)
}

for generation := 0; generation < 30; generation++ {
    if _, err := simulation.RunGeneration(context.Background()); err != nil {
        log.Fatal(err)
    }
}
```

The callbacks are where the user's domain logic lives:

```go
func myReward(step *simulation.StepContext, a *agent.Agent) (float64, *agent.Training) {
    world := step.World.(*MyWorld)
    index := step.AgentIndex(a.ID)
    if index < 0 {
        return 0, nil
    }

    // Measure any condition you like on the raw world.
    delta := 0.0
    if world.Scored(index) {
        delta = 1
    }

    // Optionally translate a target into the normalized world for backprop.
    training := &agent.Training{
        Inputs:     step.Sensors[index],
        Targets:    []float64{world.DesiredControl(index)},
        Rate:       0.05,
        Iterations: 1,
    }
    return delta, training
}
```

Returning a zero delta with a `nil` training sample is a no-op, so you can
express sparse events (scores, collisions) without special cases.

Use `simulation.InlineRunner{}` for a deterministic single-threaded reference,
and `simulation.ChannelRunner{}` for parallel evaluation. Both produce identical
results for the same seed. Set `Options.BrainFactory` if you want a brain other
than the default `NEATBrain`.

---

## 6. Gradient training

### On a NEAT genome

`agent.NEATBrain` implements both `Brain` and `Trainer`. A `Training` returned
from a reward/punish callback triggers immediate `Learn`:

```go
brain, err := agent.NewNEATBrain(genome)
player := agent.NewAgent("genome-1", brain)

// Called automatically when a callback returns a non-nil Training:
// player.Reward(delta, training)
```

Choose a trainer explicitly for different optimizers:

```go
trainer := backprop.NewTrainer(backprop.NewRMSProp(0.01, 0.9, 1e-8, 0.0, 5.0), 8)
brain, err := agent.NewNEATBrainWithTrainer(genome, trainer, backprop.MSELoss{})
```

### On a fixed topology

```go
brain, err := agent.NewBackpropBrain(
    2, 1, 4,                          // inputs, outputs, hidden
    neat.ActivationTanh,
    backprop.NewSGD(0.5, 0.9),
    10,                               // weight clip
    random.New(1),
)
```

### On a HyperNEAT substrate

```go
brain, err := agent.NewHyperNeatBrain(solver, trainer, backprop.MSELoss{})
```

### Differentiating into the CPPN

To train the CPPN weights from substrate loss:

```go
cppnTrainer, err := hyperneat.NewCppnTrainer(cppn, layout, substrateOptions, trainer, backprop.MSELoss{})
for epoch := 0; epoch < 150; epoch++ {
    for _, sample := range dataset {
        cppnTrainer.Learn(sample.Inputs, sample.Targets, 1)
    }
}
```

See `examples/hyperneat_gradient` for both paths end to end.

### Custom losses and optimizers

Implement `backprop.Loss`:

```go
type Loss interface {
    Compute(outputs, targets []float64, gradient []float64) (loss float64, err error)
}
```

or `backprop.Optimizer`:

```go
type Optimizer interface {
    Kind() string
    Update(key int, parameter, gradient float64) float64
}
```

`Kind` is used by snapshot serialization, so pick a stable string.

---

## 7. Phased searching

Phases let the search alternate between growing and shrinking structure, and
optionally exploit gradient training.

```go
options.Phases = neat.PhaseOptions{
    Enabled:         true, // cycle the schedule
    Duration:        15,   // generations per phase
    StallLimit:      8,    // advance early if fitness stalls
    TrainIterations: 2,
    Trainer: &benchmarks.SupervisedTrainer{ /* ... */ },
}
```

- `Mode` selects a fixed phase (`PhaseDefault`, `PhasePrune`,
  `PhaseBackprop`, `PhaseHybrid`) when `Enabled` is false.
- `Schedule` overrides the default `{Default, Prune}` cycle.
- `Trainer` is only required for the trainable phases. It is injected, so
  `engine/neat` never imports `engine/backprop`.

The active phase is available as `population.Phase()`.

---

## 8. HyperNEAT

```go
cppn, err := hyperneat.NewCPPN(cppnGenome) // genome with 6 inputs, >=1 output

layout := hyperneat.NewGridSubstrateLayout(1, 4, 2, 2) // bias, inputs, hidden, outputs

solverOptions := hyperneat.DefaultOptions()
solverOptions.LinkThreshold = 0.0
solverOptions.OutputActivation = neat.ActivationSigmoid

solver, err := hyperneat.NewSolver(layout, cppn, solverOptions)
outputs := make([]float64, solver.OutputCount())
solver.Activate(inputs, outputs)
```

For evolvable substrates, discover hidden points and feed them into a layout:

```go
esOptions := hyperneat.DefaultESOptions()
points, err := hyperneat.DiscoverHiddenPoints(cppn, esOptions)

layout := hyperneat.NewMappedEvolvableSubstrateLayout(4, 2)
layout.SetHiddenPoints(points)
```

The CPPN genome itself is evolved with the normal `neat.Population` machinery.
A CPPN always has `hyperneat.CPPNInputs` (6) inputs; use `cppn.OutputCount()`
for the number of outputs.

---

## 9. Snapshots and auto-save

```go
// Periodic checkpoints while training.
simulation.SetAutoSave("run_snapshot.json", 5)

// Or manually:
if err := simulation.SaveTo("run_snapshot.json"); err != nil {
    log.Fatal(err)
}

// Resume later:
restored, err := simulation.LoadSimulation("run_snapshot.json", NewMyWorld(...))
restored.SetReward(myReward) // callbacks are code, re-register them
restored.SetRunner(simulation.ChannelRunner{})
```

Snapshots include the world state, the population (genomes, species, innovation
tracker and RNG state). Resuming continues the run bit-for-bit. Population-only
snapshots are also available via `population.Encode()` and
`neat.DecodePopulation`.

For population-only state (no `World`), just write the bytes:

```go
data, err := population.Encode()
os.WriteFile("population.json", data, 0o644)
```

Add a `.gitignore` entry for `*_snapshot.json` / `*_population.json` so runtime
state is not committed.

---

## 10. Conventions

- Prefer human-readable names and the categorical package layout
  (`engine/neat`, `engine/backprop`, `engine/hyperneat`, `adapter`, `agent`,
  `simulation`).
- Never use global `math/rand`. Every stochastic component takes a seeded
  `*random.Rand`.
- Keep worlds deterministic given `Reset(seed)` so snapshots and parallel runs
  stay reproducible.
- Validate dimensions early: `NewVectorizer` and `NewSolver` return errors for
  inconsistent configuration.
- Run `gofmt`, `go vet` and the tests before committing (see the testing
  guide).
