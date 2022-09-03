package radish

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func SquareLossForward(predicted, actual *mat.Dense) float64 {
	rows, _ := predicted.Dims()
	error := 0.0
	for i := 0; i < rows; i++ {
		error += math.Pow(predicted.At(i, 0)-actual.At(i, 0), 2)
	}

	return (1.0 / float64(rows)) * error
}

func SquareLossBackward(predicted, actual *mat.Dense) *mat.Dense {
	rows, _ := predicted.Dims()

	errors := make([]float64, rows)
	for i := 0; i < rows; i++ {
		errors[i] = 2 * (predicted.At(i, 0) - actual.At(i, 0)) / float64(rows)
	}

	return mat.NewDense(rows, 1, errors)
}
