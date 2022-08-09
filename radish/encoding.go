package radish

type OneHotEncoder struct {
	labels map[string]int
	count  int
}

func NewOneHotEncoder(labels []string) *OneHotEncoder {
	tokens := tokenizeLabels(labels)

	return &OneHotEncoder{
		labels: tokens,
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

func (e *OneHotEncoder) Encode(label string) []float64 {
	vector := make([]float64, e.count)
	for i := 0; i < len(vector); i++ {
		vector[i] = 0
	}

	categoryId, found := e.labels[label]
	if found {
		vector[categoryId] = 1
	}

	return vector
}
