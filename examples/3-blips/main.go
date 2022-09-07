package main

import (
	"encoding/csv"
	"fmt"
	"github.com/vialtek/radish/radish"
	"os"
	"strconv"
)

func loadDataset() ([][]float64, []string) {
	var examples [][]float64

	csv := readCSVFile("datasets/blips/blips.csv")
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

func main() {
	encoder := radish.NewOneHotEncoder([]string{"M", "V", "N", "H"})

	examples, labels := loadDataset()
	encodedLabels := encoder.EncodeList(labels)

	model := radish.NewSequentialModel(0.01)
	model.AttachLabelEncoder(encoder)

	model.AddDenseLayer(21, 42, "relu")
	model.AddDenseLayer(42, 4, "softmax")

	model.Fit(examples, encodedLabels, 10000)
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
