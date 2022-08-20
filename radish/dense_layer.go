package radish

import (
	"gonum.org/v1/gonum/mat"
)

type DenseLayer struct {
	Weights   *mat.Dense
	Biases    *mat.Dense
	optimizer *Sgd

	forwardTensor *mat.Dense
}

func NewDenseLayer(inputs, outputs int) *DenseLayer {
	return &DenseLayer{
		Weights:   mat.NewDense(inputs, outputs, RandArray(inputs*outputs)),
		Biases:    mat.NewDense(1, outputs, RandArray(outputs)),
		optimizer: NewSgd(0.001),
	}
}

func (l *DenseLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	var output mat.Dense

	output.Mul(input, l.Weights)
	output.Add(&output, l.Biases)

	l.forwardTensor = CopyMatrix(input)
	return &output
}

func (l *DenseLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	var dL_dw, dL_dx mat.Dense

	dL_dy := CopyMatrix(input)
	dy_dw := CopyMatrix(l.forwardTensor)
	dy_dx := CopyMatrix(l.Weights)

	dL_dw.Mul(dy_dw.T(), dL_dy)

	l.optimizer.Update(l.Weights, &dL_dw)
	l.optimizer.Update(l.Biases, dL_dy)

	dL_dx.Mul(dL_dy, dy_dx.T())

	return &dL_dx
}
