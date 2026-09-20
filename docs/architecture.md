# goneat Architecture

`goneat` is a Go library for building **live simulation, training and execution
environments** around [NEAT](http://nn.cs.utexas.edu/?stanley:gecco02b) and
[HyperNEAT](https://eplex.cs.ucf.edu/hyperNEATpage/). It controls arbitrary
software through *adapters* that translate program state into a normalized
"NEAT world" and translate neural outputs back into control signals.

This document describes how the library is put together: the packages, the
core types, how data flows through an episode, and the guarantees that make
runs reproducible.

---

## 1. The normalized world

Everything a neural network sees or does is a `float64` in `[-1, 1]`.

```
arbitrary program --adapters--> normalized frames [-1,1] --channels--> brain
       ^                                                                  |
       +---------------- control frames (denormalized) <-----------------+
                           inside a lock-step Simulation episode
```

- A **sensor** reads part of the program and produces normalized values.
- A **control** consumes normalized values and applies them to the program.
- A **brain** is a pure function from input frames to output frames (plus an
  optional `Learn` for gradient training).

The library knows nothing about paddles, robots, or shapes. Only the user's
`World` and adapters do. That separation is what makes the same brain code work
across very different programs.

---

## 2. Packages and dependency graph

```
engine/vector     standalone numeric kernels (scalar + SIMD)
engine/random     standalone serializable PCG32 generator
adapter           standalone sensor/control vectorizers
engine/neat       genetic core; depends on engine/random
engine/backprop   differentiable graph; depends on engine/neat
engine/hyperneat  CPPN + substrate; depends on engine/neat, engine/backprop
agent             Brain/Trainer/Agent; depends on neat, backprop, hyperneat, random
simulation        World/Runners/Snapshot; depends on agent, neat
benchmarks        standard tasks; depends on neat, backprop, random
examples/*        runnable demos that use everything
```

The edges are deliberately acyclic and shallow. In particular `engine/neat`
never imports `engine/backprop`: gradient training is injected through a small
interface (`neat.PhaseTrainer`), so the genetic core stays independent of the
differentiable layer.

| Package | Responsibility |
|---|---|
| `engine/vector` | Allocation-free `[]float64` kernels (`Dot`, `Add`, `Sub`, `Scale`, `MulAdd`, `Sum`). SIMD when built with `GOEXPERIMENT=simd`, scalar otherwise. |
| `engine/random` | Deterministic PCG32 generator whose full state is JSON serializable, so snapshots resume exactly. |
| `adapter` | `Sensor`/`Control` interfaces and concrete adapters: enum, string similarity, 2D/3D position, scalar, bool. Also the `Vectorizer`/`Manifest` frame layout. |
| `engine/neat` | Genomes, innovation tracking, phenotype networks, mutation/crossover, speciation, population/epochs, phased searching, JSON snapshots. |
| `engine/backprop` | Differentiable graph over a `neat.Network`, losses (MSE, cross-entropy), optimizers (SGD+momentum, RMSProp), and a persistent `Trainer`. |
| `engine/hyperneat` | CPPN over a genome, substrate layouts (grid and evolvable), link-expression rules (threshold and LEO), substrate solver, ES-HyperNEAT discovery, CPPN gradient trainer. |
| `agent` | Brain-agnostic `Agent` with `Reward`/`Punish`; `NEATBrain`, `BackpropBrain`, and a HyperNEAT-substrate brain. |
| `simulation` | `World` interface, per-tick reward/punish callbacks, lock-step `InlineRunner` and `ChannelRunner`, resumable snapshots with optional auto-save. |
| `benchmarks` | Standard comparison tasks: parity, multiplexer, CartPole, polynomial-boundary classification, plus the phase-comparison harness. |

---

## 3. Episode data flow

An episode is driven by a `Runner`. Each tick is a lock-step barrier:

```
        world goroutine                          agent goroutine (one per agent)
  1. Sensors() -> one frame per agent
  2. send each frame on the agent's port -------------> receive frame
  3.                                                    Activate(frame) -> controls
  4. receive all control frames <----------------------- send controls
  5. Apply(controls); Step(dt)
  6. reward/punish callbacks run for every agent (in order)
  7. all Done? -> end episode, else next tick
```

Because step 4 waits for every agent, all agents always observe the same world
generation and every control is applied to that same generation. `InlineRunner`
runs the identical sequence sequentially, and a test asserts both runners
produce byte-identical fitness and genomes for a given seed.

After an episode, `neat.Population.Epoch` evaluates the genomes, speciates them,
and produces the next generation. When a phase is **trainable** (Backprop or
Hybrid), offspring are gradient-trained after reproduction and before the next
episode.

---

## 4. `engine/neat` - the genetic core

### Genes and genomes

```go
type NodeGene struct {
    ID         int
    Type       NodeType        // Input, Bias, Hidden, Output
    Activation ActivationType  // linear, sigmoid, tanh, relu, gaussian, sin, cos, abs, mult, add
}

type ConnectionGene struct {
    Innovation int
    In, Out    int
    Weight     float64
    Enabled    bool
}

type Genome struct {
    ID          int
    Nodes       []NodeGene
    Connections []ConnectionGene
    Fitness     float64
    SpeciesID   int
    Generation  int
}
```

Node ID convention: inputs are `0..nIn-1`, the bias is `nIn`, outputs are
`nIn+1..nIn+nOut`, and hidden nodes are allocated upward from there. The
`InnovationTracker` assigns stable innovation numbers to connections so
crossover can align genes from different genomes.

### Phenotype

`Network` topologically sorts a genome and evaluates it. Aggregation
activations (`mult`, `add`) combine their inputs specially; everything else is
a weighted sum followed by the node's activation. `BreakCycles` guarantees a
genome is feed-forward after crossover, so every genome is evaluable and
differentiable.

### Population and speciation

`Population` owns the genomes, species, innovation tracker, RNG and phase
controller. `Epoch` does:

1. evaluate every genome (`Evaluator` interface),
2. cluster into `Species` by compatibility distance,
3. update species statistics and stagnation,
4. adjust the compatibility threshold toward the target species count,
5. reproduce (fitness sharing, elitism, crossover, mutation),
6. advance the search phase if needed.

`GenerationStats` and an optional `Observer` expose per-generation progress.

### Phased searching

`Phase` selects the mutation strategy for a generation:

| Phase | Behaviour |
|---|---|
| `PhaseDefault` | complexify: weight mutation + add link/node + toggle |
| `PhasePrune` | simplify: weight mutation + remove link/node |
| `PhaseBackprop` | exploit: weight mutation only; offspring are gradient-trained |
| `PhaseHybrid` | complexify + gradient-train offspring |

`PhaseOptions` either fixes a `Mode` or cycles a `Schedule` (default
`{Default, Prune}`), advancing after `Duration` generations or early when the
best fitness has stalled for `StallLimit` generations. Gradient training is
provided through the injected `PhaseTrainer`, which keeps `engine/neat`
independent of `engine/backprop`.

### Serialization

`Population` implements `json.Marshaler`/`json.Unmarshaler`. A snapshot includes
the options, generation counters, the innovation tracker, the genome list,
species membership, the best-ever genome, **and the RNG state**. Combined with
`engine/random`, a restored population continues a run bit-for-bit.

---

## 5. `engine/backprop` - gradient training on evolved topologies

`Graph` builds a differentiable evaluation from a `neat.Network`:

- `Forward(inputs)` caches pre- and post-activations,
- `Backward(loss, targets)` computes `dLoss/dWeight` per connection innovation,
- `AccumulateOutputGradient(gradient)` seeds an output gradient without
  clearing weight gradients, which lets a function that queries the network
  many times (the CPPN case) accumulate parameter gradients.

`Trainer` applies gradients to `Genome.Connections[].Weight` with a persistent
optimizer state keyed by innovation, then clips weights. `Loss` is an interface
with `MSELoss` and `CrossEntropyLoss`. Optimizers are `SGD` (with momentum) and
`RMSProp` (with decay and gradient clipping); both serialize their state.

Because the genome is the single source of truth, genetic operators, gradient
training and JSON snapshots all operate on the same weights.

---

## 6. `engine/hyperneat` - indirect encoding

A CPPN is a NEAT network evaluated on coordinates:

```go
type CPPN struct{ /* wraps a neat.Genome */ }
func (c *CPPN) Query(source, target Point) ([]float64, error) // 6 coords -> outputs
```

A `SubstrateLayout` places neurons in the unit hypercube. `GridSubstrateLayout`
puts inputs on `y=-1`, outputs on `y=+1` and the bias at the origin;
`MappedEvolvableSubstrateLayout` holds hidden points discovered by ES-HyperNEAT.

`NewSolver(layout, cppn, options)` builds a substrate genome: for every pair of
connected neuron groups it queries the CPPN and turns the output into a weight.
A link is expressed when `|output| >= LinkThreshold` (normalized into
`WeightRange`), or when a second CPPN output (`UseLEO`) is positive.
`DiscoverHiddenPoints` runs a variance-driven quadtree and returns stable-region
centres as hidden neurons.

There are two gradient paths over HyperNEAT:

- `agent.NewHyperNeatBrain(solver, ...)` trains the **substrate's** connection
  weights with ordinary backprop.
- `hyperneat.NewCppnTrainer(cppn, layout, options, ...)` differentiates the
  substrate loss back into the **CPPN** that generated it, using the chain rule
  through the link-expression function and `Graph.AccumulateOutputGradient`.

---

## 7. `agent` - brains and fitness

```go
type Brain interface {
    Activate(inputs, outputs []float64) error
    Clone() Brain
    Encode() ([]byte, error)
    Decode(data []byte) error
}

type Trainer interface {
    Learn(inputs, targets []float64, loss backprop.Loss, rate float64, iterations int) error
}

type Agent struct {
    ID      string
    Alive   bool
    Fitness float64
    Brain   Brain
}
```

`Agent` is brain-agnostic. `Reward(delta, training)` and `Punish(delta,
training)` adjust fitness and, when a training sample is supplied and the brain
implements `Trainer`, perform immediate gradient learning. `NEATBrain` wraps a
`neat.Genome`; `BackpropBrain` builds a fixed hidden-layer topology;
`NewHyperNeatBrain` wraps a substrate solver.

---

## 8. `simulation` - connecting brains to a world

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

`Simulation` combines a `World`, a `neat.Population`, a `Runner`, and the
user's callbacks:

```go
type RewardFunc func(step *StepContext, agent *agent.Agent) (delta float64, training *agent.Training)
type PunishFunc func(step *StepContext, agent *agent.Agent) (delta float64, training *agent.Training)
```

Per tick the simulation invokes `Reward` and then `Punish` for every agent. The
callbacks see the raw `World` and the normalized frames, so the user decides
what counts as reward and how to translate a domain condition into a training
target. `RunGeneration` plays an episode and then evolves the population.

### Snapshots

`Simulation.Snapshot` writes a versioned JSON document containing the world
state, the full population (including RNG), and the run configuration.
`SaveTo`/`LoadSimulation` persist and restore it. `Options.AutoSavePath` and
`AutoSaveEvery` write a checkpoint every *N* generations automatically.
Reward/Punish callbacks are code and are re-registered by the caller after a
restore.

---

## 9. `adapter` - translation

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

`Vectorizer` concatenates sensors and controls into flat frames and produces a
`Manifest` describing each slot (name, offset, dimensions, and optional
HyperNEAT placement). Concrete adapters: `EnumAdapter` (ordinal or one-hot),
`StringAdapter` (anchor-dictionary similarity), `Position2D`/`Position3D`,
`ScalarAdapter` (clamp or tanh), `BoolAdapter`.

---

## 10. `engine/random` and `engine/vector`

`engine/random` is a PCG32 generator with `Uint64`, `Intn`, `Float64`,
`NormFloat64` (Box-Muller with a cached spare) and JSON round-tripping. Every
stochastic component in the library takes a `*random.Rand`; there is no global
`math/rand`.

`engine/vector` provides the numeric kernels. With `GOEXPERIMENT=simd` the same
functions compile to the portable `simd` package; otherwise a scalar unrolled
implementation is used. Both backends are covered by the same reference tests.

---

## 11. Determinism guarantees

- No global randomness: each population, world and agent owns a seeded
  `*random.Rand`; `engine/random` makes its state part of snapshots.
- The lock-step barrier means scheduling cannot change results.
- Map iteration never affects results: innovation/weight accumulation and
  speciaton use sorted order.
- `InlineRunner` and `ChannelRunner` are asserted to be byte-identical.
- Population and simulation snapshots restore bit-for-bit.
