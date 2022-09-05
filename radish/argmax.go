package radish

import (
	"gonum.org/v1/gonum/mat"
)

func argMaxArray(vector []float64) int {
	mat := mat.NewDense(len(vector), 1, vector)
	return argMax(mat)
}

func argMax(vector *mat.Dense) int {
	r, c := vector.Dims()
	if c > 1 {
		// TODO: Should panic
		return -1
	}

	highestIndex := 0
	highestValue := vector.At(0, 0)

	for i := 0; i < r; i++ {
		if vector.At(i, 0) > highestValue {
			highestIndex = i
			highestValue = vector.At(i, 0)
		}
	}

	return highestIndex
}
