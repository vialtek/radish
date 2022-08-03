package main

import (
	"github.com/vialtek/radish/radish"
)

func main() {
	trainExamples := [][]float64{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}}
	// TODO: encode labels to result vector
	trainLabels := [][]float64{{0}, {1}, {1}, {0}}

	model := radish.NewSequentialModel()
	model.AddLayer(radish.NewDenseLayer(3, 1, "tanh"))

	model.Fit(trainExamples, trainLabels, 5000)
	//model.Train(trainExamples[0], trainLabels[0])

	modelOutput := model.Evaluate([]float64{1, 0, 0})
	radish.PrintMatrix(modelOutput, "Output (1)")

	modelOutput2 := model.Evaluate([]float64{0, 1, 1})
	radish.PrintMatrix(modelOutput2, "Output (0)")
}
