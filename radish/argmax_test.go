package radish

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestArgMaxCorrect(t *testing.T) {
	vec := mat.NewDense(4, 1, []float64{1, 2, 1, 1})

	highestIndex := argMax(vec)
	if highestIndex != 1 {
		t.Error("argMax is returning wrong index", highestIndex, 1)
	}
}

func TestArgMaxWrongDimension(t *testing.T) {
	vec := mat.NewDense(2, 2, []float64{1, 2, 1, 1})

	highestIndex := argMax(vec)
	if highestIndex != -1 {
		t.Error("argMax should not work on matrixes of more than one row", highestIndex, -1)
	}
}

func TestRowsColumnsMissmatch(t *testing.T) {
	vec := mat.NewDense(1, 4, []float64{1, 2, 1, 1})

	highestIndex := argMax(vec)
	if highestIndex != -1 {
		t.Error("argMax should not work with column matrix", highestIndex, -1)
	}
}

func TestArgMaxArray(t *testing.T) {
	vec := []float64{0.2, 0.5, 0.98, 0.2}

	highestIndex := argMaxArray(vec)
	if highestIndex != 2 {
		t.Error("argMaxArray is returning wrong index", highestIndex, 2)
	}
}
