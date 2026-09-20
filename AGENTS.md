# AGENTS.md

## Project

`goneat` is a Go library for building live simulation / training / execution
environments around NeuroEvolution of Augmenting Topologies (NEAT) and
HyperNEAT. It controls arbitrary software through *adapters* that vectorize
arbitrary inputs into the `[-1, 1]` "NEAT world" and translate neural outputs
back into the controlled program.

## Toolchain

- Go 1.27 (`go 1.27` in `go.mod`).
- Standard library only. No third-party dependencies.
- JSON serialization uses the stable `encoding/json` package.
- SIMD vector operations use the portable `simd` package behind
  `GOEXPERIMENT=simd`; a scalar fallback is used otherwise.

## Commands

Always run these before considering a task done:

```sh
make fmt        # go fmt ./...
make vet        # go vet ./...
make test       # go test ./...
make test-race  # go test -race ./...
make test-simd  # GOEXPERIMENT=simd go test ./...
make bench      # benchmarks across all packages
make bench-simd # benchmarks across all packages (SIMD)
make bench-xor  # training correctness/efficiency (loss + optimizer comparison)
make bench-phases # phase comparison over standard benchmarks (slow)
```

There is no external linter configured; `gofmt` + `go vet` are the gate.

## Conventions

- Human-readable names for variables, functions, types and packages.
- Categorical packages: `engine/neat`, `engine/backprop`, `engine/hyperneat`,
  `engine/vector`, `adapter`, `agent`, `simulation`.
- No global `math/rand`: every stochastic component takes a seeded
  `*rand.Rand` so runs are reproducible.
- Determinism: `InlineRunner` and `ChannelRunner` must produce identical
  results for the same seed.
- Every feature is tracked in `docs/TASKS.md` with implemented / tested /
  verified (benchmark) status. Keep `TODO.md` updated incrementally.
- `engine/backprop/xor_*_test.go` is the training proof/regression harness:
  deterministic XOR training tests plus benchmarks reporting
  `iters/op`, `final_loss` and `solve_rate`. Run `make bench-xor` before and
  after training-related changes and compare.
- HyperNEAT supports two gradient paths: `agent.NewHyperNeatBrain` trains
  substrate weights; `hyperneat.CppnTrainer` differentiates substrate loss back
  into the CPPN.
- Phased Searching lives in `engine/neat/phase.go`: phases Default
  (complexify), Prune (simplify), Backprop and Hybrid. `Options.Phases` selects
  a fixed mode or a cycling schedule; `PhaseTrainer` is injected so
  `engine/neat` never imports `engine/backprop`.
- Standard comparison tasks and the phase benchmark live in `benchmarks/`.
- Examples auto-save runtime state to the working directory as
  `*_snapshot.json` / `*_population.json`; these are git-ignored, including via
  per-example `.gitignore` files. Do not commit them.
- Do not commit generated coverage files or `references/`.

## Layout

```
engine/vector     SIMD/scalar math kernels
engine/neat       genetic core (genome, network, species, population)
engine/backprop   differentiable graph over a neat.Genome
engine/hyperneat  CPPN, substrate layouts, ES-HyperNEAT
adapter           sensor/control vectorizers and placement manifest
agent             Agent, Brain, Trainer, Training
simulation        World, Simulation, callbacks, runners, snapshot
examples/         xor, pong, hyperneat, hyperneat_gradient, parabola
benchmarks/       parity, multiplexer, cartpole, polynomial, phase comparison
docs/             plan, roadmap, architecture, task tracking
```
