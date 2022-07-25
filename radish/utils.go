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

func RandArray(elementCount int) []float64 {
	rand.Seed(time.Now().UnixNano())

	randArray := make([]float64, elementCount)
	for i := range randArray {
		randArray[i] = rand.NormFloat64()
	}

	return randArray
}