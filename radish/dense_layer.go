package radish

import (
	"gonum.org/v1/gonum/mat"
)

type DenseLayer struct {
	activation string
	Weights    *mat.Dense
	Biases     *mat.Dense
	optimizer  *Sgd

	forwardTensor *mat.Dense
}

func NewDenseLayer(inputs, outputs int, activation string) *DenseLayer {
	return &DenseLayer{
		activation: activation,
		Weights:    mat.NewDense(inputs, outputs, RandArray(inputs*outputs)),
		Biases:     mat.NewDense(1, outputs, RandArray(outputs)),
		optimizer:  NewSgd(0.001),
	}
}

func (l *DenseLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	var output mat.Dense

	output.Mul(input, l.Weights)
	output.Add(&output, l.Biases)
	output.Apply(l.activationForward, &output)

	l.forwardTensor = CopyMatrix(input)
	return &output
}

// FIXME: activation is not accounted (also for biases)
func (l *DenseLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	var dL_dw, dL_dx mat.Dense

	dL_dy := CopyMatrix(input)
	dy_dw := CopyMatrix(l.forwardTensor)
	dy_dx := CopyMatrix(l.Weights)

	dL_dw.Mul(dy_dw.T(), dL_dy)
	//PrintMatrix(dL_dy, "dl_dy")
	//PrintMatrix(&dL_dw, "dl_dw")
	//PrintMatrix(l.forwardTensor, "Forward sensor")
	l.optimizer.Update(l.Weights, &dL_dw)
	l.optimizer.Update(l.Biases, dL_dy)

	dL_dx.Mul(dL_dy, dy_dx.T())

	return &dL_dx
}
