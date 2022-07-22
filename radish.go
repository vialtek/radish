package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"

	"github.com/vialtek/radish/radish"
)

func main() {
	model := radish.NewSequentialModel()

	model.AddLayer(radish.NewDenseLayer(4, 4, "relu"))
	model.AddLayer(radish.NewDenseLayer(4, 2, "relu"))

	inputData := mat.NewDense(1, 4, []float64{1, 1, 0, 0})
	outputData := model.ForwardProp(inputData)

	formatedOutput := mat.Formatted(outputData, mat.Prefix("     "), mat.Squeeze())
	fmt.Printf("Result: %v\n\n", formatedOutput)
}
