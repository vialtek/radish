package radish

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type layer interface {
	ForwardProp(input *mat.Dense) *mat.Dense
	BackwardProp(input *mat.Dense) *mat.Dense
}

type SequentialModel struct {
	layers []layer

	optimizer    *Sgd
	labelEncoder *OneHotEncoder
}

func NewSequentialModel(learningRate float64) *SequentialModel {
	return &SequentialModel{
		optimizer: NewSgd(learningRate),
	}
}

func (m *SequentialModel) AttachLabelEncoder(labelEncoder *OneHotEncoder) {
	m.labelEncoder = labelEncoder
}

func (m *SequentialModel) AddDenseLayer(inputs, outputs int, activation string) {
	denseLayer := NewDenseLayer(inputs, outputs, m.optimizer)
	m.layers = append(m.layers, denseLayer)

	activationLayer := NewActivationLayer(activation)
	m.layers = append(m.layers, activationLayer)
}

func (m *SequentialModel) Evaluate(input []float64) *mat.Dense {
	curY := mat.NewDense(len(input), 1, input)

	for _, layer := range m.layers {
		curY = layer.ForwardProp(curY)
	}

	return curY
}

func (m *SequentialModel) Train(examples [][]float64, labels [][]float64) float64 {
	totalError := 0.0
  errorVectors := make([][]float64, len(examples))

  // 1. Acumulate loss across batch
	for i, example := range examples {
		outcome := m.Evaluate(example)
		actual := mat.NewDense(len(labels[i]), 1, labels[i])

		error := SquareLossForward(outcome, actual)
		totalError += error

		errorVectors[i] = SquareLossBackward(outcome, actual)
	}

	// 2. Calculate mean loss
	meanErrorGradient := ZeroArray(len(errorVectors[0]))
	for _, errorVector := range errorVectors {
		for j := 0; j < len(errorVector); j++ {
			meanErrorGradient[j] += errorVector[j]
		}
	}

	for i := 0; i < len(meanErrorGradient); i++ {
		meanErrorGradient[i] = 1.0 / float64(len(meanErrorGradient)) * meanErrorGradient[i]
	}

	// 3. Propagate error gradient
	lossGradient := mat.NewDense(len(meanErrorGradient), 1, meanErrorGradient)
	for i := len(m.layers) - 1; i >= 0; i-- {
		lossGradient = m.layers[i].BackwardProp(lossGradient)
	}

	return totalError
}

func (m *SequentialModel) ResultToLabel(output *mat.Dense) string {
	if m.labelEncoder == nil {
		return ""
	}

	return m.labelEncoder.IndexToLabel(argMax(output))
}

func (m *SequentialModel) Fit(examples [][]float64, labels [][]float64, batchSize int, epochs int) {
	batch := NewMinibatch(examples, labels, batchSize)

	for epoch := 1; epoch <= epochs; epoch++ {
		batch.Rewind()
		epochError := 0.0

		for batch.HasNext(){
			batchExamples, batchLabels := batch.Next()
			epochError += m.Train(batchExamples, batchLabels)
		}

		meanError := 1.0 / float64(len(examples)) * epochError
		fmt.Println("Epoch: ", epoch, "/", epochs, "Error: ", meanError)

		if m.labelEncoder != nil {
			fmt.Println("Accuary: ", m.AccuracyOfEpoch(examples, labels)*100, "%")
		}
	}
}

func (m *SequentialModel) AccuracyOfEpoch(examples [][]float64, labels [][]float64) float64 {
	corrCount := 0
	for i, example := range examples {
		result := m.Evaluate(example)
		if argMax(result) == argMaxArray(labels[i]) {
			corrCount += 1
		}
	}

	return float64(corrCount) / float64(len(examples))
}
