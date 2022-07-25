package radish

import (
	"gonum.org/v1/gonum/mat"
)

type DenseLayer struct {
	activation string
	Weights    *mat.Dense
	Biases     *mat.Dense

	forwardIn   mat.Dense
	forwardOut  mat.Dense
	backwardIn  mat.Dense
	backwardOut mat.Dense
}

func NewDenseLayer(inputs, outputs int, activation string) *DenseLayer {
	return &DenseLayer{
		activation: activation,
		Weights:    mat.NewDense(inputs, outputs, RandArray(inputs*outputs)),
		Biases:     mat.NewDense(1, outputs, RandArray(outputs)),
	}
}

func (l *DenseLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	l.forwardIn.Copy(input)
	var output mat.Dense

	output.Mul(input, l.Weights)
	output.Add(&output, l.Biases)
	output.Apply(l.activationForward, &output)

	l.forwardOut.Copy(&output)
	return &output
}

// FIXME: activation is not accounted
// TODO: do we need backwardIn, backwardOut and forwardOut?
//       ... maybe just store forward tensor?
func (l *DenseLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	var dL_dy, dy_dw, dy_dx, dL_dw, dL_dx mat.Dense

	l.backwardIn.Copy(input)

	dL_dy.Copy(&l.backwardIn)
	dy_dw.Copy(&l.forwardIn)
	dy_dx.Copy(l.Weights)

	dL_dw.Mul(dy_dw.T(), &dL_dy)
	// TODO: update weights here

	dL_dx.Mul(&dL_dy, dy_dx.T())

	l.backwardOut.Copy(&dL_dx)
	return &dL_dx
}
