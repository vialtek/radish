package radish

import (
	"gonum.org/v1/gonum/mat"
)

type optimizer interface {
	Update(weights, change *mat.Dense)
}

func NewOptimizer(learningRate float64) optimizer {
	return &Sgd{
		learningRate: learningRate,
	}
}

type Sgd struct {
	learningRate float64
}

func (o *Sgd) Update(weights, change *mat.Dense) {
	rows, cols := change.Dims()

	rateArr := make([]float64, rows*cols)
	for i := 0; i < rows*cols; i++ {
		rateArr[i] = o.learningRate
	}

	rateMatrix := mat.NewDense(rows, cols, rateArr)

	var changeMatrix mat.Dense
	changeMatrix.MulElem(change, rateMatrix)
	weights.Sub(weights, &changeMatrix)
}
