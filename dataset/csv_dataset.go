package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

//
// Expected format is CSV file separated by commas.
// Label is in first row, rest is example
//

func LoadCSVDataset(filePath string) ([][]float64, []string) {
	var examples [][]float64

	csv := readCSVFile(filePath)
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
