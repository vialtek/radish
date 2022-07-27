package radish

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func SquareLossForward(predicted, actual *mat.Dense) []float64 {
	_, columns := predicted.Dims()
	errors := make([]float64, columns)

	for i := 0; i < columns; i++ {
		errors[i] = math.Pow(predicted.RawRowView(0)[i]-actual.RawRowView(0)[i], 2)
	}

	return errors
}
