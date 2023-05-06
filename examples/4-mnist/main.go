package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/vialtek/radish/radish"
)

func main() {
	examples, labels := loadDataset()

	encoder := radish.NewOneHotEncoder([]string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
	encodedLabels := encoder.EncodeList(labels)

	model := radish.NewSequentialModel(0.01, "square")
	model.AttachLabelEncoder(encoder)

	model.AddDenseLayer(784, 100, "relu")
	model.AddDenseLayer(100, 10, "softmax")

	model.Fit(examples, encodedLabels, 16, 20)
}

func loadDataset() ([][]float64, []string) {
	var examples [][]float64

	csv := readCSVFile("datasets/mnist/mnist_train.csv")
	labels := make([]string, len(csv))

	for i := range csv {
		labels[i] = csv[i][0]

		example := make([]float64, len(csv[i])-1)
		for j := 1; j < len(csv[i]); j++ {
			example[j-1], _ = strconv.ParseFloat(csv[i][j], 64)
		}

		examples = append(examples, example)
	}

	return examples, labels
}

func readCSVFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		fmt.Println(err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		fmt.Println("Unable to parse a CSV file", err)
	}

	return records
}
