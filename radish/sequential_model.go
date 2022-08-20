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
}

func NewSequentialModel() *SequentialModel {
	return &SequentialModel{}
}

func (m *SequentialModel) AddLayer(inputs, outputs int, activation string) {
	denseLayer := NewDenseLayer(inputs, outputs)
	m.layers = append(m.layers, denseLayer)

	activationLayer := NewActivationLayer(activation)
	m.layers = append(m.layers, activationLayer)
}

func (m *SequentialModel) Evaluate(input []float64) *mat.Dense {
	curY := mat.NewDense(1, len(input), input)

	for _, layer := range m.layers {
		curY = layer.ForwardProp(curY)
	}

	return curY
}

func (m *SequentialModel) Train(example []float64, labels []float64) {
	outcome := m.Evaluate(example)
	actual := mat.NewDense(1, len(labels), labels)

	curGrad := SquareLossBackward(outcome, actual)
	// Iterate backwards
	for i := len(m.layers) - 1; i >= 0; i-- {
		curGrad = m.layers[i].BackwardProp(curGrad)
	}
}

func (m *SequentialModel) Fit(examples [][]float64, labels [][]float64, epochs int) {
	for epoch := 1; epoch <= epochs; epoch++ {
		fmt.Println("Epoch: ", epoch, "/", epochs)
		for i, example := range examples {
			m.Train(example, labels[i])
		}
	}
}
