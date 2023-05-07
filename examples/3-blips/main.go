package main

import (
	"github.com/vialtek/radish/dataset"
	"github.com/vialtek/radish/radish"
)

func main() {
	encoder := radish.NewOneHotEncoder([]string{"M", "V", "N", "H"})

	examples, labels := dataset.LoadCSVDataset("data/blips/blips.csv")
	encodedLabels := encoder.EncodeList(labels)

	model := radish.NewSequentialModel(0.01, "crossentropy")
	model.AttachLabelEncoder(encoder)

	model.AddDenseLayer(21, 42, "relu")
	model.AddDenseLayer(42, 4, "softmax")

	model.Fit(examples, encodedLabels, 16, 10000)
}
