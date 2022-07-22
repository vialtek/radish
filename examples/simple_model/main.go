package main

import (
	"github.com/vialtek/radish/radish"
)

func main() {
	model := radish.NewSequentialModel()

	model.AddLayer(radish.NewDenseLayer(4, 4, "relu"))
	model.AddLayer(radish.NewDenseLayer(4, 2, "relu"))

	modelOutput := model.Evaluate([]float64{1, 1, 0, 0})
	radish.PrintMatrix(modelOutput, "Output")
}
