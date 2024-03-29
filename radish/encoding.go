package radish

type OneHotEncoder struct {
	labels map[string]int
	keys   []string
	count  int
}

func NewOneHotEncoder(labels []string) *OneHotEncoder {
	tokens := tokenizeLabels(labels)

	return &OneHotEncoder{
		labels: tokens,
		keys:   labelsArray(tokens),
		count:  len(tokens),
	}
}

func tokenizeLabels(labels []string) map[string]int {
	uniqueCount := 0
	tokenTable := make(map[string]int)

	for _, val := range labels {
		if _, seenValue := tokenTable[val]; !seenValue {
			tokenTable[val] = uniqueCount
			uniqueCount += 1
		}
	}

	return tokenTable
}

func labelsArray(labels map[string]int) []string {
	keys := make([]string, len(labels))

	for k, v := range labels {
		keys[v] = k
	}

	return keys
}

func (e *OneHotEncoder) Encode(label string) []float64 {
	vector := ZeroArray(e.count)

	categoryId, found := e.labels[label]
	if found {
		vector[categoryId] = 1
	}

	return vector
}

func (e *OneHotEncoder) EncodeList(labels []string) [][]float64 {
	vector := make([][]float64, len(labels))

	for i, val := range labels {
		vector[i] = e.Encode(val)
	}

	return vector
}

func (e *OneHotEncoder) IndexToLabel(index int) string {
	return e.keys[index]
}
