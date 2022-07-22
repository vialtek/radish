package radish

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func PrintMatrix(matrix *mat.Dense, label string) {
	formatted := mat.Formatted(matrix, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v: \n%v\n\n", label, formatted)
}
