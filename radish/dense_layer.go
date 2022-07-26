package radish

import (
	"gonum.org/v1/gonum/mat"
)

type DenseLayer struct {
	activation string
	Weights    *mat.Dense
	Biases     *mat.Dense
	optimizer  *Sgd

	storedTensor mat.Dense
}

func NewDenseLayer(inputs, outputs int, activation string) *DenseLayer {
	return &DenseLayer{
		activation: activation,
		Weights:    mat.NewDense(inputs, outputs, RandArray(inputs*outputs)),
		Biases:     mat.NewDense(1, outputs, RandArray(outputs)),
		optimizer:  NewSgd(),
	}
}

func (l *DenseLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	var output mat.Dense

	output.Mul(input, l.Weights)
	output.Add(&output, l.Biases)
	output.Apply(l.activationForward, &output)

	l.storedTensor.Copy(input)
	return &output
}

// FIXME: activation is not accounted
func (l *DenseLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	var dL_dy, dy_dw, dy_dx, dL_dw, dL_dx mat.Dense

	dL_dy.Copy(input)
	dy_dw.Copy(&l.storedTensor)
	dy_dx.Copy(l.Weights)

	dL_dw.Mul(dy_dw.T(), &dL_dy)
	l.optimizer.Update(l.Weights, &dL_dw)

	dL_dx.Mul(&dL_dy, dy_dx.T())

	return &dL_dx
}
