package radish

import (
	"gonum.org/v1/gonum/mat"
	"testing"
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

func TestSoftmaxForward(t *testing.T) {
	softmaxLayer := NewActivationLayer("softmax")

	inputTensor := mat.NewDense(1, 3, []float64{0.7, 0.99, 1.2})
	outputTensor := softmaxLayer.ForwardProp(inputTensor)

	sum := 0.0
	for _, val := range outputTensor.RawRowView(0) {
		sum += val
	}

	if sum > 1.0 || sum < 0.9999 {
		t.Error("Sum of softmax didn't returned 1", outputTensor, sum)
	}
}

func TestUnknownActivationFunction(t *testing.T) {
	unknownActivationLayer := NewActivationLayer("NOTEXISTS")

	inputVector := []float64{-1, 0.1, 4}
	inputTensor := mat.NewDense(1, 3, inputVector)
	outputTensor := unknownActivationLayer.ForwardProp(inputTensor)
	outputVector := outputTensor.RawRowView(0)

	if !vectorEquals(inputVector, outputVector) {
		t.Error("When using unknow activaiton layer, identity should be returned", inputVector, outputVector)
	}
}
