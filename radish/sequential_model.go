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

func (m *SequentialModel) AddLayer(newLayer layer) {
	m.layers = append(m.layers, newLayer)
}

func (m *SequentialModel) Evaluate(input []float64) *mat.Dense {
	curY := mat.NewDense(1, len(input), input)

	for _, layer := range m.layers {
		curY = layer.ForwardProp(curY)
	}

	return curY
}

func (m *SequentialModel) Train(example []float64, label float64) {
	outcome := m.Evaluate(example)
	actual := mat.NewDense(1, 1, []float64{1})

	error := SquareLossForward(outcome, actual)
	fmt.Println("Error: ", error)
}
