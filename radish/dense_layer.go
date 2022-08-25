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

func (l *DenseLayer) BackwardProp(outputGradient *mat.Dense) *mat.Dense {
	var weightsGradient, inputGradient mat.Dense

	weightsGradient.Mul(outputGradient, l.forwardTensor.T())
	inputGradient.Mul(l.Weights.T(), outputGradient)

	l.optimizer.Update(l.Weights, &weightsGradient)
	l.optimizer.Update(l.Biases, outputGradient)

	return &inputGradient
}
