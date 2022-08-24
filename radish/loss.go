package radish

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func SquareLossForward(predicted, actual *mat.Dense) float64 {
	_, columns := predicted.Dims()
	error := 0.0
	for i := 0; i < columns; i++ {
		error += math.Pow(predicted.RawRowView(0)[i]-actual.RawRowView(0)[i], 2)
	}

	return (1.0 / float64(columns)) * error
}

func SquareLossBackward(predicted, actual *mat.Dense) *mat.Dense {
	_, columns := predicted.Dims()
	errors := make([]float64, columns)

	for i := 0; i < columns; i++ {
		errors[i] = 2 * (predicted.RawRowView(0)[i] - actual.RawRowView(0)[i]) / float64(columns)
	}

	return mat.NewDense(1, columns, errors)
}
