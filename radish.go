package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"

	"github.com/vialtek/radish/radish"
)

func main() {
	a := mat.NewDense(2, 2, []float64{
		4, 0,
		0, 4,
	})

	b := mat.NewDense(2, 2, []float64{
		2, 2,
		2, 2,
	})

	var c mat.Dense
	c.Mul(a, b)

	fc := mat.Formatted(&c, mat.Prefix("     "), mat.Squeeze())
	fmt.Printf("Mul two matrixes:\n c = %v\n\n", fc)

	model := radish.NewSequentialModel()

	model.AddLayer(radish.NewDenseLayer(4, 4, "relu"))
	model.AddLayer(radish.NewDenseLayer(4, 2, "relu"))

	inputData := mat.NewDense(1, 4, []float64{1, 1, 0, 0})
	fmt.Println(model.ForwardProp(inputData))
}
