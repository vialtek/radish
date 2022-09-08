package radish

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func PrintMatrix(matrix *mat.Dense, label string) {
	formatted := mat.Formatted(matrix, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v: \n%v\n\n", label, formatted)
}

func CopyMatrix(matrix *mat.Dense) *mat.Dense {
	rows, columns := matrix.Dims()
	newMatrix := mat.NewDense(rows, columns, nil)
	newMatrix.Copy(matrix)

	return newMatrix
}

func IdentityMatrix(size int) *mat.Dense {
	identityMatrix := mat.NewDense(size, size, nil)
	for i := 0; i < size; i++ {
		identityMatrix.Set(i, i, 1)
	}

	return identityMatrix
}

func TileMatrix(input *mat.Dense, times int) *mat.Dense {
	var output mat.Dense

	if times <= 1 {
		return input
	}

	for i := 1; i < times; i++ {
		if i == 1 {
			output.Augment(input, input)
		} else {
			var tmp mat.Dense

			tmp.Augment(&output, input)
			output = tmp
		}
	}

	return &output
}

func ArrayFromMatrix(input *mat.Dense) []float64 {
	var array []float64
	r, c := input.Dims()

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			array = append(array, input.At(i, j))
		}
	}

	return array
}

func RandArray(elementCount int) []float64 {
	rand.Seed(time.Now().UnixNano())

	randArray := make([]float64, elementCount)
	for i := range randArray {
		randArray[i] = rand.NormFloat64()
	}

	return randArray
}

func ZeroArray(elementCount int) []float64 {
	randArray := make([]float64, elementCount)
	for i := range randArray {
		randArray[i] = 0.0
	}

	return randArray
}
