package radish

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

func NewActivationLayer(activation string) layer {
	switch activation {
	case "relu":
		return &ReluActivationLayer{}
	case "tanh":
		return &TanhActivationLayer{}
	case "sigmoid":
		return &SigmoidActivationLayer{}
	case "softmax":
		return &SoftmaxActivationLayer{}
	default:
		return &IdentityActivationLayer{}
	}
}

type ReluActivationLayer struct {
	forwardTensor *mat.Dense
}

func relu(elem float64) float64 {
	if elem < 0 {
		return 0
	}

	return elem
}

func reluPrime(elem float64) float64 {
	if elem > 0 {
		return 1
	}

	return 0
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

	output.MulElem(input, d_forward)
	return &output
}

type TanhActivationLayer struct {
	forwardTensor *mat.Dense
}

func tanh(elem float64) float64 {
	return math.Tanh(elem)
}

func tanhPrime(elem float64) float64 {
	return 1 - math.Pow(math.Tanh(elem), 2)
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

	output.MulElem(input, d_forward)
	return &output
}

type SigmoidActivationLayer struct {
	forwardTensor *mat.Dense
}

func sigmoid(elem float64) float64 {
	return 1 / (1 + math.Exp(-elem))
}

func sigmoidPrime(elem float64) float64 {
	s := sigmoid(elem)
	return s * (1 - s)
}

func (l *SigmoidActivationLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	l.forwardTensor = CopyMatrix(input)
	output := CopyMatrix(input)
	rows, cols := output.Dims()

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			output.Set(i, j, sigmoid(input.At(i, j)))
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
			d_forward.Set(i, j, sigmoidPrime(l.forwardTensor.At(i, j)))
		}
	}

	output.MulElem(input, d_forward)
	return &output
}

type SoftmaxActivationLayer struct {
	forwardTensor *mat.Dense
	outputTensor  *mat.Dense
}

func (l *SoftmaxActivationLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	var output *mat.Dense

	l.forwardTensor = CopyMatrix(input)
	output = CopyMatrix(input)

	sum := 0.0
	rows, cols := output.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			newVal := math.Exp(output.At(i, j))
			output.Set(i, j, newVal)
			sum += newVal
		}
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			output.Set(i, j, output.At(i, j)/sum)
		}
	}

	l.outputTensor = CopyMatrix(output)

	return output
}

func (l *SoftmaxActivationLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	var tmpMatrix, output mat.Dense
	n, _ := l.outputTensor.Dims()

	tiled := TileMatrix(l.outputTensor, n)
	identity := IdentityMatrix(n)

	tmpMatrix.Sub(identity, tiled.T())
	tmpMatrix.MulElem(tiled, &tmpMatrix)

	output.Mul(&tmpMatrix, input)

	return &output
}

type IdentityActivationLayer struct{}

func (l *IdentityActivationLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	return input
}

func (l *IdentityActivationLayer) BackwardProp(input *mat.Dense) *mat.Dense {
	return input
}
