GO ?= go
GOEXPERIMENT ?= simd

.PHONY: all fmt vet test test-race test-simd bench bench-simd bench-xor bench-phases bench-vector cover tidy clean

all: fmt vet test

fmt:
	$(GO) fmt ./...

vet:
	$(GO) vet ./...

test:
	$(GO) test ./...

test-race:
	$(GO) test -race ./...

# The SIMD path lives behind GOEXPERIMENT=simd (Go 1.27 portable simd package).
test-simd:
	GOEXPERIMENT=$(GOEXPERIMENT) $(GO) test ./...

bench:
	$(GO) test -bench=. -benchmem ./...

bench-simd:
	GOEXPERIMENT=$(GOEXPERIMENT) $(GO) test -bench=. -benchmem ./...

# Training correctness + efficiency. Reports ns/op, iters/op, final_loss and
# solve_rate for each loss/optimizer pairing, plus the evolutionary XOR run and
# the HyperNEAT substrate/CPPN gradient steps. Run repeatedly and compare with
# `go test -bench='XOR|TrainStep' -count=5` to track whether changes improve or
# worsen training.
bench-xor:
	$(GO) test -bench='XOR|TrainStep' -benchmem -benchtime=1s ./engine/backprop/ ./engine/hyperneat/ ./examples/xor/

bench-vector:
	$(GO) test -bench=. -benchmem ./engine/vector/...

# Phase comparison across standard benchmarks (parity, multiplexer, cartpole).
# Reports generations/op, solve_rate and best_fitness per strategy. This runs
# full searches, so it takes minutes; use -benchtime to control repetition.
bench-phases:
	$(GO) test -bench='PhaseComparison' -benchtime=1x -run '^$$' ./benchmarks/

cover:
	$(GO) test -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out

tidy:
	$(GO) mod tidy

clean:
	$(GO) clean -testcache
