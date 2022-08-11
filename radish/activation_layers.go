package radish

import (
	"gonum.org/v1/gonum/mat"
)

func NewActivationLayer(activation string) layer {
	switch activation {
	case "relu":
		return &ReluActivationLayer{}
	case "tanh":
		return &TanhActivationLayer{}
	case "sigmoid":
		return &SigmoidActivationLayer{}
	default:
		return &IdentityActivationLayer{}
	}
}

type ReluActivationLayer struct {
	forwardTensor *mat.Dense
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
			d_forward.Set(i, j, reluPrime(l.forwardTensor.At(i, j)))
		}
	}

	output.Mul(input, d_forward)
	return &output
}

type TanhActivationLayer struct {
	forwardTensor *mat.Dense
}

func (l *TanhActivationLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	l.forwardTensor = CopyMatrix(input)
	output := CopyMatrix(input)
	rows, cols := output.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			output.Set(i, j, tanh(input.At(i, j)))
		}
	}

	return output
}

func (l *TanhActivationLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	var output mat.Dense
	d_forward := CopyMatrix(l.forwardTensor)

	rows, cols := l.forwardTensor.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			d_forward.Set(i, j, tanhPrime(l.forwardTensor.At(i, j)))
		}
	}

	output.Mul(input, d_forward)
	return &output
}

type SigmoidActivationLayer struct {
	forwardTensor *mat.Dense
}

func (l *SigmoidActivationLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	l.forwardTensor = CopyMatrix(input)
	output := CopyMatrix(input)
	rows, cols := output.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			output.Set(i, j, tanh(input.At(i, j)))
		}
	}

	return output
}

func (l *SigmoidActivationLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	var output mat.Dense
	d_forward := CopyMatrix(l.forwardTensor)

	rows, cols := l.forwardTensor.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			d_forward.Set(i, j, tanhPrime(l.forwardTensor.At(i, j)))
		}
	}

	output.Mul(input, d_forward)
	return &output
}

type IdentityActivationLayer struct{}

func (l *IdentityActivationLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	return input
}

func (l *IdentityActivationLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	return input
}
