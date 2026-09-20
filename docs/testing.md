# goneat Testing and Benchmarking Guide

`goneat` treats correctness and reproducibility as first-class. The test suite
proves that training works and that runs are deterministic; the benchmark suite
measures how efficiently different strategies train.

---

## 1. Running the suites

```sh
make fmt          # gofmt
make vet          # go vet
make test         # go test ./...
make test-race    # go test -race ./...
make test-simd    # GOEXPERIMENT=simd go test ./...
make bench        # all benchmarks
make bench-xor    # training correctness/efficiency
make bench-phases # phase comparison over standard tasks (slow)
make bench-vector # vector kernel benchmarks (scalar + SIMD)
make cover        # coverage profile
```

`gofmt`, `go vet` and `go test` are the quality gate. There is no external
linter.

---

## 2. Test layout

| Package | Focus |
|---|---|
| `adapter` | encode/decode round-trips, clipping, manifest offsets |
| `agent` | frame activation, reward-triggered learning, clone independence, encode/decode |
| `engine/vector` | scalar/SIMD results match a naive reference for many lengths |
| `engine/random` | deterministic sequences, distribution sanity, JSON resume |
| `engine/neat` | genes, networks, mutation, crossover, distance, speciation, phases, snapshots |
| `engine/backprop` | numerical gradient check, learns XOR, optimizer state |
| `engine/hyperneat` | CPPN symmetry, substrate construction, CPPN gradient check |
| `simulation` | runner equivalence, callbacks, auto-save and resume |
| `benchmarks` | standard tasks and phase comparison |
| `examples/*` | end-to-end demos |

Run one package or one test:

```sh
go test ./engine/neat/
go test -run TestPopulationSolvesXOR -v ./engine/neat/
go test -run 'TestCppnTrainer' -v ./engine/hyperneat/
```

---

## 3. Correctness proofs

### Training actually works: the XOR harness

`engine/backprop/xor_training_test.go` is the canonical proof that gradient
training works. It trains a small `2 -> 4(tanh) -> 1(sigmoid)` network on XOR,
which is not linearly separable and therefore cannot be solved without a hidden
layer.

- `TestXORTrainingConvergesAcrossSeeds` runs two loss/optimizer pairings
  (`MSE + SGD`, `CrossEntropy + RMSProp`) over ten seeds and asserts every run
  solves XOR and reduces loss.
- `TestXORTrainingIsDeterministic` runs the same configuration twice and asserts
  bit-identical weights, iterations and loss.
- `TestXORTrainingRegressionBound` pins a canonical seed and asserts it is
  solved within a bounded number of passes and final loss. This is the
  regression guard: a change that slows or degrades training trips it.
- `TestXORLossFunctionsReduceLoss` compares losses at an equal training budget.

### Gradient math: numerical checks

`engine/backprop/backprop_test.go` compares analytic gradients against central
finite differences for every connection of a small genome. `engine/hyperneat`
does the same for CPPN gradients (`TestCppnTrainerGradientDirection`), which
validates the chain rule through the substrate and link-expression function.

### Structural correctness

- `TestCrossoverAndMutationStayAcyclic` fuzzes 1000 crossover+mutate rounds and
  asserts every child is evaluable, covering the cycle-breaking safety net.
- `engine/hyperneat` verifies that an even CPPN produces identical weights for a
  mirrored substrate, and that ES-HyperNEAT discovers hidden points inside the
  hypercube.
- `benchmarks` proves a solved polynomial-boundary champion uses hidden nodes
  with **multiple distinct activation functions**.

### Determinism and reproducibility

- `engine/random` asserts sequences are seed-stable and that a JSON snapshot
  resumes exactly (including the cached Gaussian sample).
- `simulation.TestRunnersProduceIdenticalResults` asserts `InlineRunner` and
  `ChannelRunner` produce byte-identical genomes and world state.
- `simulation.TestSnapshotResumeMatchesUninterrupted` asserts a run paused,
  snapshotted and resumed matches an uninterrupted run.
- `simulation.TestAutoSaveAndLoad` asserts periodic checkpoints are written and
  can be loaded and continued.
- `engine/neat.TestPopulationSnapshotResumeIsDeterministic` does the same at the
  population level.

---

## 4. Benchmarks

Benchmarks use the standard Go `testing.B` mechanism and report both `ns/op`
and custom quality metrics.

### Vector kernels

```sh
make bench-vector
make bench-simd            # same, with GOEXPERIMENT=simd
```

At n = 1024 on an AMD Ryzen 7 7840HS:

| kernel | scalar | SIMD |
|---|---|---|
| `Dot` | 447.6 ns/op | 155.4 ns/op |
| `Add` | 533.6 ns/op | 222.0 ns/op |
| `Sub` | 363.6 ns/op | 206.7 ns/op |
| `Scale` | 256.3 ns/op | 146.0 ns/op |
| `MulAdd` | 409.9 ns/op | 215.3 ns/op |
| `Sum` | 163.9 ns/op | 93.2 ns/op |

### Training efficiency and quality

```sh
make bench-xor
```

This reports:

- `BenchmarkXORLearnStep` - cost of one gradient step.
- `BenchmarkXORConvergence/<loss>` - `ns/op` (time to solve), plus custom
  metrics `iters/op`, `final_loss`, `solve_rate`.
- `BenchmarkXORFixedBudget/<loss>` - equal-cost comparison reporting
  `final_loss` and `accuracy`.
- `BenchmarkPopulationSolvesXOR` - evolutionary search, reporting
  `generations/op` and `solve_rate`.
- `BenchmarkSubstrateTrainStep` / `BenchmarkCppnTrainStep` - HyperNEAT gradient
  step costs.

Custom metrics are read as: lower `iters/op`/`generations/op` is faster
convergence, higher `solve_rate` is more reliable, lower `final_loss` is a
better fit.

### Comparing search strategies over time

```sh
make bench-phases
```

`BenchmarkPhaseComparison` runs every standard task against every strategy:

- tasks: `parity2`, `parity3`, `parity4`, `mux6`, `cartpole`, `cubic`,
  `quintic`, `septic`
- strategies: `default`, `prune`, `backprop`, `hybrid`, `phased`

Each sub-benchmark reports `generations/op`, `solve_rate` and `best_fitness`.
Because full searches are run, this target takes a few minutes; use
`-benchtime` to raise the number of repetitions.

To compare a change before and after, save the output and diff:

```sh
go test -bench='XOR|TrainStep' -count=5 ./... | tee before.txt
# make a change
go test -bench='XOR|TrainStep' -count=5 ./... | tee after.txt
diff before.txt after.txt
```

---

## 5. Standard benchmark tasks

`benchmarks/` implements a `Task` interface:

```go
type Task interface {
    Name() string
    InputCount() int
    OutputCount() int
    Evaluate(genome *neat.Genome) (float64, error)
    Solved(genome *neat.Genome) bool
    Trainer() neat.PhaseTrainer
}
```

| Task | Shape | Why |
|---|---|---|
| `NewParity(bits)` | `bits -> 1` | combinatorial, standard NEAT benchmark |
| `NewMultiplexer(addressBits)` | `(a + 2^a) -> 1` | classic NEAT/HyperNEAT benchmark |
| `NewCartPole(maxSteps)` | `4 -> 1` | sequential control, multi-start, heuristic trainer |
| `NewCubicBoundary(res)` | `2 -> 1` | curved `y = x^3` boundary |
| `NewQuinticBoundary(res)` / `NewSepticBoundary(res)` | `2 -> 1` | curved `y = x^5` / `x^7` boundary (square grid) |
| `NewOddPowerBoundary(name, power)` | `2 -> 1` | odd-power boundary on the tuned 11x9 grid used by the comparison |

`NewOddPowerBoundary` samples a denser x axis than y axis. On a coarse square
grid `x^5` and `x^7` produce identical labels, because the two curves only
differ inside a thin band that no sample lands in; the extra x resolution makes
them distinct without the cost of a fine square grid.

`RunSearch(DefaultRunConfig(task, strategy, seed, maxGenerations))` runs a
strategy and returns `RunResult{Generations, Solved, BestFitness, NodeCount,
Elapsed, Champion}`.

The polynomial task is the clearest demonstration of per-node activation
functions: a solved champion must contain hidden nodes with several distinct
activations.

---

## 6. Writing new tests

- Prefer table-driven tests and deterministic seeds.
- For randomness, construct `random.New(seed)`; never use `math/rand`.
- Assert determinism where it matters: run twice and compare serialized output.
- When adding a feature, add a test that would fail if the feature regressed.
- When a test encodes a reproducible outcome, treat the bound as a regression
  guard and document the seed and budget in a comment.

## 7. Writing new benchmarks

- Benchmark a stable workload; seed any randomness.
- Report quality with `b.ReportMetric(value, "unit")` after `b.StopTimer()`.
- Use `b.ReportAllocs()` and `b.ResetTimer()` correctly.
- For search comparisons, accumulate over several seeds inside the loop and
  report averages, so a single lucky seed does not dominate.
