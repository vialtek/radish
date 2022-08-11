package radish

import (
	"testing"
	"gonum.org/v1/gonum/mat"
)

func TestReluActivationForward(t *testing.T) {
	reluLayer := NewActivationLayer("relu")

	inputTensor := mat.NewDense(1, 3, []float64{-1, 0.1, 4})
	outputTensor := reluLayer.ForwardProp(inputTensor)

	expectedVector := []float64{0, 0.1, 4}
	outputVector := outputTensor.RawRowView(0)

	if !vectorEquals(expectedVector, outputVector) {
		t.Error("Relu activation layer do not return correct vector", expectedVector, outputVector)
	}
}
