package radish

import (
	"gonum.org/v1/gonum/mat"
)

type layer interface {
	ForwardProp(input *mat.Dense) *mat.Dense
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
