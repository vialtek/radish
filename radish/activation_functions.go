package radish

import (
	"math"
)

func sigmoid(elem float64) float64 {
	return 1 / (1 + math.Exp(-elem))
}

func sigmoidPrime(elem float64) float64 {
	s := sigmoid(elem)
	return s * (1 - s)
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
