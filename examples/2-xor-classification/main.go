package main

import (
	"fmt"
	"github.com/vialtek/radish/radish"
)

func main() {
	encoder := radish.NewOneHotEncoder([]string{"0", "1"})

	trainExamples := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	trainLabels := []string{"0", "1", "1", "0"}
	encodedLabels := encoder.EncodeList(trainLabels)

	model := radish.NewSequentialModel(0.01)
	model.AttachLabelEncoder(encoder)

	model.AddDenseLayer(2, 3, "relu")
	model.AddDenseLayer(3, 2, "softmax")

	model.Fit(trainExamples, encodedLabels, 10000)

	modelOutput := model.Evaluate([]float64{1, 0})
	radish.PrintMatrix(modelOutput, "Model output")
	label := model.ResultToLabel(modelOutput)
	fmt.Println("Model label (should be 1): ", label)
}
