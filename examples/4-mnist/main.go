package main

import (
	"github.com/vialtek/radish/dataset"
	"github.com/vialtek/radish/radish"
)

func main() {
	examples, labels := dataset.LoadCSVDataset("data/mnist/mnist_train.csv")

	encoder := radish.NewOneHotEncoder([]string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
	encodedLabels := encoder.EncodeList(labels)

	model := radish.NewSequentialModel(0.01, "square")
	model.AttachLabelEncoder(encoder)

	model.AddDenseLayer(784, 100, "relu")
	model.AddDenseLayer(100, 10, "softmax")

	model.Fit(examples, encodedLabels, 16, 20)
}
