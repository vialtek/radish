package radish

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestFlattenLayerForward(t *testing.T) {
	// [1 3]
	// [2 4]
	twoByTwo := mat.NewDense(2, 2, []float64{1, 2, 3, 4})

	flattenLayer := NewFlattenLayer()
	output := flattenLayer.ForwardProp(twoByTwo)

	r, c := output.Dims()

	if c > 1 {
		t.Error("Flatten should get rid of colums.", r)
	}

	if r != 4 {
		t.Error("Flatten should put all the items into one row.", c, 4)
	}
}

func TestFlattenLayerBackward(t *testing.T) {
	// [1 3]
	// [2 4]
	inputVector := []float64{1, 2, 3, 4}
	twoByTwo := mat.NewDense(2, 2, inputVector)

	flattenLayer := NewFlattenLayer()
	output := flattenLayer.ForwardProp(twoByTwo)
	backwardOutput := flattenLayer.BackwardProp(output)

	ir, ic := twoByTwo.Dims()
	or, oc := backwardOutput.Dims()

	if ir != or && ic != oc {
		t.Error("Input and output shape should be same", ir, ic, or, oc)
	}

	outputVector := []float64{1, 2, 3, 4}
	if !vectorEquals(inputVector, outputVector) {
		t.Error("Input and output vectors are not in same order", inputVector, outputVector)
	}
}
