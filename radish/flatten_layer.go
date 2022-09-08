package radish

import (
	"gonum.org/v1/gonum/mat"
)

type FlattenLayer struct {
	forwardTensor *mat.Dense
}

func NewFlattenLayer() *FlattenLayer {
	return &FlattenLayer{}
}

func (l *FlattenLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	if l.forwardTensor == nil {
		l.forwardTensor = CopyMatrix(input)
	}

	vector := ArrayFromMatrix(input)
	return mat.NewDense(len(vector), 1, vector)
}

func (l *FlattenLayer) BackwardProp(output *mat.Dense) *mat.Dense {
	r, c := l.forwardTensor.Dims()

	return mat.NewDense(r, c, ArrayFromMatrix(output))
}
