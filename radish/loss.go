package radish

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

func MeanSquaredErrorLoss(predicted, actual *mat.Dense) float64 {
	var sumSquareError float64 = 0.0
	_, columns := predicted.Dims()

	for i := 0; i < columns; i++ {
		sumSquareError += math.Pow(predicted.RawRowView(0)[i]-actual.RawRowView(0)[i], 2)
	}

	if sumSquareError == 0 {
		return 0
	}

	meanSquareError := 1.0 / (float64(columns) * sumSquareError)
	return meanSquareError
}
