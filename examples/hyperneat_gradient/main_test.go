package main

import "testing"

func TestSubstrateTrainingReducesLoss(t *testing.T) {
	before, after, err := runSubstrateTraining(1, 200)
	if err != nil {
		t.Fatalf("runSubstrateTraining: %v", err)
	}
	if after >= before {
		t.Fatalf("substrate training did not reduce loss: %.6f -> %.6f", before, after)
	}
	if after > before*0.1 {
		t.Fatalf("substrate training reduced loss insufficiently: %.6f -> %.6f", before, after)
	}
}

func TestCPPNTrainingReducesLoss(t *testing.T) {
	before, after, err := runCPPNTraining(2, 150)
	if err != nil {
		t.Fatalf("runCPPNTraining: %v", err)
	}
	if after >= before {
		t.Fatalf("cppn training did not reduce loss: %.6f -> %.6f", before, after)
	}
}
