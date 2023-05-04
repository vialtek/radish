package radish

import (
	"testing"
)

func TestEvenMinibatch(t *testing.T) {
	examples := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	labels := [][]float64{{0}, {1}, {1}, {0}, {1}}

	batch := NewMinibatch(examples, labels, 2)

	count := 0
	for batch.HasNext() {
		samples, _ := batch.Next()

		if len(samples) != 2 {
			t.Error("Batch size should be 2, is ", len(samples))
		}
		count += 1
	}

	if count != 2 {
		t.Error("There should be two iterators of minibatch, there are ", count)
	}
}

func TestOddMinibatch(t *testing.T) {
	examples := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {6, 3}}
	labels := [][]float64{{0}, {1}, {1}, {0}, {1}}

	batch := NewMinibatch(examples, labels, 4)

	samples, _ := batch.Next()
	if len(samples) != 4 {
		t.Error("Minibatch should return 4 samples, it did returned ", len(samples))
	}

	if !batch.HasNext() {
		t.Error("There should be next batch available")
	}

	secondSamples, _ := batch.Next()
	if len(secondSamples) != 1 {
		t.Error("Minibatch should return 1 last sample, it did returned ", len(secondSamples))
	}

	if batch.HasNext() {
		t.Error("There should be no next batch available")
	}
}

func TestFullBatch(t *testing.T) {
	examples := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {6, 3}}
	labels := [][]float64{{0}, {1}, {1}, {0}, {1}}

	batch := NewMinibatch(examples, labels, -1)

	samples, _ := batch.Next()
	if len(examples) != len(samples) {
		t.Error("Minibatch with batchSize -1 should return all samples, but returned: ", len(examples))
	}

	if batch.HasNext() {
		t.Error("There should be no next batch available")
	}
}

func TestRewindBatch(t *testing.T) {
	examples := [][]float64{{9, 1}, {2, 3}, {5, 6}, {7, 8}}
	labels := [][]float64{{0}, {1}, {1}, {0}, {1}}

	batch := NewMinibatch(examples, labels, 2)
	batch.Next()
	samples, _ := batch.Next()

	if !vectorEquals(samples[0], []float64{5, 6}) {
		t.Error("Batch returned unexpected result, it should return {5, 6} and it returned ", samples[0])
	}

	if batch.HasNext() {
		t.Error("There should be no next batch available")
	}

	batch.Rewind()
	rewindSamples, _ := batch.Next()

	if !vectorEquals(rewindSamples[0], []float64{9, 1}) {
		t.Error("Batch returned unexpected result, it should return {9, 1} and it returned ", rewindSamples[0])
	}

	if !batch.HasNext() {
		t.Error("There should be next batch available")
	}
}
