package radish

import (
	"math"
)

func (l *DenseLayer) activationElem(i, j int, elem float64) float64 {
	switch l.activation {
	case "relu":
		return relu(elem)
	case "sigmoid":
		return sigmoid(elem)
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
