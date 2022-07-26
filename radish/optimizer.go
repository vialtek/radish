package radish

import (
	"gonum.org/v1/gonum/mat"
)

type optimizer interface {
	Update(weights, change *mat.Dense)
}

type Sgd struct {
	learningRate float64
}

func NewSgd() *Sgd {
	return &Sgd{
		learningRate: 0.00001,
	}
}

func (o *Sgd) Update(weights, change *mat.Dense) {
	// TODO: update the weights
}
