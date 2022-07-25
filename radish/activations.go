package radish

import (
	"math"
)

func (l *DenseLayer) activationForward(i, j int, elem float64) float64 {
	switch l.activation {
	case "relu":
		return relu(elem)
	case "sigmoid":
		return sigmoid(elem)
	case "tanh":
		return tanh(elem)
	default:
		return elem
	}
}

func (l *DenseLayer) activationBackward(i, j int, elem float64) float64 {
	switch l.activation {
	case "relu":
		return reluPrime(elem)
	case "sigmoid":
		return sigmoidPrime(elem)
	case "tanh":
		return tanhPrime(elem)
	default:
		return elem
	}
}

func sigmoid(elem float64) float64 {
	return 1 / (1 + math.Exp(-elem))
}

func sigmoidPrime(elem float64) float64 {
	return math.Exp(-elem) / (math.Pow(1+math.Exp(-elem), 2))
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

func tanh(elem float64) float64 {
	return math.Tanh(elem)
}

func tanhPrime(elem float64) float64 {
	return 1 - math.Pow(math.Tanh(elem), 2)
}
