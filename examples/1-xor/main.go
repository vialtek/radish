package main

import (
	"github.com/vialtek/radish/radish"
)

func main() {
	trainExamples := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainLabels := [][]float64{{0}, {1}, {1}, {0}}

	model := radish.NewSequentialModel(0.01)
	model.AddDenseLayer(2, 3, "relu")
	model.AddDenseLayer(3, 1, "tanh")

	model.Fit(trainExamples, trainLabels, 2, 10000)

	modelOutput := model.Evaluate([]float64{1, 0})
	radish.PrintMatrix(modelOutput, "Output (should be 1)")
}
