package radish

type Minibatch struct {
	examples  [][]float64
	labels    [][]float64
	batchSize int

	currentBatch int
}

func NewMinibatch(examples [][]float64, labels [][]float64, batchSize int) *Minibatch {
	// batchSize -1 => we are not spliting dataset into minibatches
	if batchSize == -1 {
		batchSize = len(examples)
	}

	return &Minibatch{
		examples:     examples,
		labels:       labels,
		batchSize:    batchSize,
		currentBatch: 0,
	}
}

func (b *Minibatch) batchCount() int {
	batchCount := len(b.examples) / b.batchSize
	if len(b.examples)%b.batchSize > 0 {
		batchCount += 1
	}

	return batchCount
}

func (b *Minibatch) Rewind() {
	b.currentBatch = 0
}

func (b *Minibatch) HasNext() bool {
	return b.batchCount() > b.currentBatch
}

func (b *Minibatch) Next() ([][]float64, [][]float64) {
	startIndex := b.currentBatch * b.batchSize
	stopIndex := startIndex + b.batchSize

	if stopIndex > len(b.examples) {
		stopIndex = len(b.examples)
	}

	examples := b.examples[startIndex:stopIndex]
	labels := b.labels[startIndex:stopIndex]

	b.currentBatch += 1

	return examples, labels
}
