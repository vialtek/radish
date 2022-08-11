package radish

import (
	"gonum.org/v1/gonum/mat"
)

type ReluActivationLayer struct {
	forwardTensor *mat.Dense
}

func NewActivationLayer(activation string) layer {
	return &ReluActivationLayer{}
}

func (l *ReluActivationLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	l.forwardTensor = CopyMatrix(input)
	output := CopyMatrix(input)
	rows, cols := output.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			output.Set(i, j, relu(input.At(i, j)))
		}
	}

	return output
}

func (l *ReluActivationLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	var output mat.Dense
	d_forward := CopyMatrix(l.forwardTensor)

	rows, cols := l.forwardTensor.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			d_forward.Set(i, j, reluPrime(l.forwardTensor.At(i,j)))
		}
	}

	output.Mul(input, d_forward)
	return &output
}
