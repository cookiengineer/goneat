# goneat

`goneat` is a Go decision engine: a library for building **live simulation,
training and execution environments** around [NEAT](http://nn.cs.utexas.edu/?stanley:gecco02b)
and [HyperNEAT](https://eplex.cs.ucf.edu/hyperNEATpage/).

It controls arbitrary software through adapters that translate program state
into a normalized "NEAT world" and translate neural outputs back into control
signals:

```
arbitrary program --adapters--> normalized frames [-1,1] --channels--> brain
       ^                                                                  |
       +---------------- controls (denormalized) <------------------------+
                          inside a lock-step Simulation episode
```

Because the engine only ever sees `float64` values in `[-1, 1]`, the same
training code drives a Pong paddle, a robot, a classifier, or anything else you
can express as sensors and controls.

## What is in the box

- **`engine/neat`** - genetic core: genomes, innovation tracking, phenotype
  networks, mutation, crossover, speciation, population/epochs, and phased
  searching. JSON snapshots include RNG state, so runs resume exactly.
- **`engine/backprop`** - a differentiable graph over an evolved genome, with
  MSE/cross-entropy losses and SGD+momentum / RMSProp optimizers. NEAT grows
  the topology; gradient descent refines the weights.
- **`engine/hyperneat`** - CPPN-based indirect encoding, grid and evolvable
  substrate layouts, threshold/LEO link expression, ES-HyperNEAT hidden-node
  discovery, and CPPN gradient training.
- **`agent`** - a brain-agnostic `Agent` whose `Reward`/`Punish` callbacks can
  trigger immediate gradient learning.
- **`simulation`** - a `World` interface, user-defined reward/punish callbacks,
  a lock-step `ChannelRunner` (one goroutine per agent) and a deterministic
  `InlineRunner`, plus resumable snapshots with optional auto-save.
- **`adapter`** - enums, string-dictionary similarity, 2D/3D positions,
  scalars, booleans, and the `Vectorizer`/`Manifest` frame layout.
- **`engine/vector`** - scalar and `GOEXPERIMENT=simd` math kernels.
- **`engine/random`** - a serializable PCG32 generator for reproducible runs.

## Quick start

```sh
go get github.com/cookiengineer/goneat

go run ./examples/xor                  # evolve a NEAT solution to XOR
go run ./examples/parabola             # learn a curved decision boundary
go run ./examples/pong                 # evolve a paddle controller
go run ./examples/hyperneat            # ES-HyperNEAT shape detection
go run ./examples/hyperneat_gradient   # gradient training of substrate + CPPN
```

Minimal training loop:

```go
options := neat.DefaultOptions(4, 2)     // 4 sensors -> 2 controls
options.PopulationSize = 200
population, _ := neat.NewPopulation(options)

world := NewMyWorld(200)
sim, _ := simulation.NewSimulation(world, population, simulation.Options{
    Ticks:  600,
    DT:     1.0 / 60.0,
    Runner: simulation.ChannelRunner{},
    Reward: myReward,
})
sim.SetAutoSave("run_snapshot.json", 5)  // checkpoint to the working directory
sim.RunGenerations(context.Background(), 30)
```

## Benchmark results

Numbers below were produced with `make bench-xor`, `make bench-phases` and
`make bench-vector` on an AMD Ryzen 7 7840HS (16 threads), Go 1.27.
Re-run them on your machine; treat them as relative comparisons rather than
absolute constants.

### Vector kernels (n = 1024)

| kernel | scalar | `GOEXPERIMENT=simd` |
|---|---|---|
| `Dot` | 447.6 ns/op | 155.4 ns/op |
| `Add` | 533.6 ns/op | 222.0 ns/op |
| `Sub` | 363.6 ns/op | 206.7 ns/op |
| `Scale` | 256.3 ns/op | 146.0 ns/op |
| `MulAdd` | 409.9 ns/op | 215.3 ns/op |
| `Sum` | 163.9 ns/op | 93.2 ns/op |

The SIMD backend is roughly 2-3x faster and produces results equivalent to the
scalar backend within floating-point tolerance.

### Gradient training (XOR, `2 -> 4(tanh) -> 1(sigmoid)`)

`make bench-xor` measures both speed and training quality.

| benchmark | ns/op | iters/op | final_loss | solve_rate |
|---|---|---|---|---|
| `XORConvergence/mse-sgd` | 131,448 | 15.8 | 0.1042 | 1.000 |
| `XORConvergence/cross-entropy-rmsprop` | 161,382 | 18.7 | 0.4191 | 1.000 |
| `XORLearnStep` | 2,250 | - | - | - |
| `PopulationSolvesXOR` | 17.8 ms | 12 generations | - | 1.000 |
| `SubstrateTrainStep` (HyperNEAT) | 678 | - | - | - |
| `CppnTrainStep` (HyperNEAT) | 6,867 | - | - | - |

At an equal budget (200 passes) both losses solve XOR:

| loss | final_loss | accuracy |
|---|---|---|
| MSE + SGD | 0.0001466 | 1.000 |
| Cross-entropy + RMSProp | 0.0000000 | 1.000 |

### Search strategy comparison

`make bench-phases` runs every standard task against the four phase modes plus
the alternating phased search. Values are **generations to solve**; `no` means
"not solved within budget", with the best fitness in parentheses.

| task | default | prune | backprop | hybrid | phased |
|---|---|---|---|---|---|
| parity2 (2->1) | 11 | no | no | **4** | 11 |
| parity3 (3->1) | 86 | no | no | **34** | no |
| parity4 (4->1) | no (0.909) | no (0.800) | no (0.800) | no (0.941) | no (**0.947**) |
| multiplexer 6 (6->1) | no (0.859) | no (0.841) | no (0.829) | no (**0.923**) | no (0.852) |
| cartpole (4->1) | 2 | 3 | 41 | 5 | 2 |
| cubic `y = x^3` (2->1) | 134 | no (0.953) | no (0.952) | **116** | 131 |

Seed 1, population 200, `-benchtime=1x`. Lower generations is better; for
unsolved tasks, higher best fitness is better.

The comparison makes the trade-offs concrete:

- **Prune-only and Backprop-only cannot solve the combinatorial or curved
  tasks** because they cannot add the hidden units those tasks require. This is
  expected, not a bug: only complexifying phases can grow structure.
- **Hybrid usually converges fastest** on tasks that need structure, because
  gradient training refines the weights of newly grown topology.
- **Phased (complexify/simplify cycles)** matches default on easy tasks and can
  reach the best fitness on tasks that reward pruning, but simplification can
  slow early convergence.

### Curved decision boundary demo

`go run ./examples/parabola` learns to classify 2D points against `y = x^3`.
A straight line cannot separate the classes, so the search must grow hidden
neurons and use several activation functions. The demo prints the result:

```
solved: true | accuracy 1.0000 | hidden nodes: 20 | distinct activations: 6
activation histogram: sigmoid:1 tanh:2 relu:2 sin:4 abs:5 mult:6
```

This demonstrates per-node activation functions and structural expansion
working together.

## Testing

```sh
make test        # unit + integration tests
make test-race   # race detector
make test-simd   # tests built with GOEXPERIMENT=simd
make bench-xor   # training correctness + efficiency
make bench-phases# search strategy comparison (slow)
```

The suite is the proof that training works and that runs are reproducible:
numerical gradient checks, deterministic XOR convergence across seeds, a
seed-pinned regression bound, `InlineRunner`/`ChannelRunner` equivalence, and
snapshot/resume bit-identity.

## Documentation

- [Architecture](docs/architecture.md) - packages, data flow, core types.
- [Implementation guide](docs/implementation.md) - build a world, wire a
  simulation, add training, HyperNEAT, snapshots.
- [Testing and benchmarking guide](docs/testing.md) - correctness proofs,
  benchmarks, how to compare strategies.

## Repository layout

```
engine/vector     SIMD/scalar math kernels
engine/random     serializable PCG32 generator
engine/neat       genetic core (genome, network, species, population)
engine/backprop   differentiable graph over a neat.Genome
engine/hyperneat  CPPN, substrate layouts, ES-HyperNEAT
adapter           sensor/control vectorizers and placement manifest
agent             Agent, Brain, Trainer, Training
simulation        World, Simulation, callbacks, runners, snapshot
examples/         xor, parabola, pong, hyperneat, hyperneat_gradient
benchmarks/       parity, multiplexer, cartpole, polynomial, phase comparison
docs/             architecture, implementation, testing
```

The `references/` directory contains third-party research code used only as a
design reference and is excluded from the build.

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt).

