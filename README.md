# GoNEAT Decision Engine

This projects evolved out of the learnings from the previous [lycheejs](https://github.com/cookiengineer/lycheejs)
Engine Project.

It is a HyperNEAT based decision engine, made for embedding into other Go programs
to allow meta-learning, live training and evaluation, and the automated evolution
of adaptive neural network graphs.

Due to HyperNEAT and NEAT only requiring one generic input matrix and a bias node,
this can be freely combined with other sensors that normalized their output into
vector space (from `-1.0` to `+1.0`).

## Prerequisites

Reading Material, heavily recommended:

- [NEAT](http://nn.cs.utexas.edu/?stanley:gecco02b)
- [HyperNEAT](https://eplex.cs.ucf.edu/hyperNEATpage/)

## Motivation

The GoNEAT decision engine controls arbitrary software through adapters that translate
program state into a normalized "NEAT world" and translate neural outputs back into
control signals for the rest of the program.

```
program --adapters--> normalized frames --channels--> brain
   ^                                                    |
   +------------ controls (denormalized) <--------------+
```

Because the engine only ever sees `float64` values in `[-1, 1]`, the same
training code drives a Pong paddle, a robot, a classifier, or anything else you
can express as sensors and controls.

## Features

- `engine/neat` implements genomes, innovation tracking, phenotype
  networks, mutation, crossover, speciation, population/epochs, and phased
  searching.
- `engine/backprop` is the differentiable graph over an evolved genome, with
  MSE/cross-entropy losses and SGD+momentum / RMSProp optimizers. NEAT grows
  the topology in epoches whereas gradient descent refines the weights.
- `engine/hyperneat` is the CPPN-based indirect encoding, grid and evolvable
  substrate layouts, threshold/LEO link expression, ES-HyperNEAT hidden-node
  discovery, and CPPN gradient training.
- `agent` is a brain-agnostic `Agent` whose `Reward` / `Punish` callbacks can
  trigger immediate gradient learning.
- `simulation` contains `World` interface, user-defined reward / punish callbacks,
  a `ChannelRunner` (one goroutine per agent) and a deterministic `InlineRunner`,
  and resumable snapshots with optional auto-save.
- `adapter` contains adapters for Go's native data types. Currently this is
  `bool`, `int`, `uint`, `float32`, `float64`, `byte`, `rune`, `complex`, `time.Duration`,
  slices, maps, strings, enums, and 2D/3D positions.
- `engine/vector` contains the scalar and SIMD math kernels.
- `engine/random` contains a serializable PCG32 generator for reproducible runs.

## Usage

```sh
go get github.com/cookiengineer/goneat

go run ./examples/xor;                  # evolve a NEAT solution to XOR
go run ./examples/parabola;             # learn a curved decision boundary
go run ./examples/pong;                 # evolve a paddle controller
go run ./examples/hyperneat;            # ES-HyperNEAT shape detection
go run ./examples/hyperneat_gradient;   # gradient training of substrate + CPPN
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

## Benchmarking

`GOEXPERIMENT=simd` activates the SIMD kernel, which performs roughly 2-3x faster
than the defaulted Scalar kernel. Go 1.27 or later required.

```
make bench-xor;
make bench-phases;
make bench-vector;
```

## Testing

```sh
make test;         # unit + integration tests
make test-race;    # race detector
make test-simd;    # tests built with GOEXPERIMENT=simd
make bench-xor;    # training correctness + efficiency
make bench-phases; # search strategy comparison (slow)
```

The suite is the proof that training works and that runs are reproducible.
It contains numerical gradient checks, deterministic XOR convergence across
seeds, a seed-pinned regression bound, `InlineRunner` / `ChannelRunner`
equivalence test, and a snapshot / resume bit-identity test.

## Documentation

- [Architecture](docs/architecture.md) documents packages, data flow, core types.
- [Implementation guide](docs/implementation.md) documents how to use this library.
- [Testing and benchmarking guide](docs/testing.md) documents the correctness proofs and benchmarks.

## License

This project is licensed under the [MIT License](LICENSE.txt).

