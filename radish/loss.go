package radish

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type lossFunction interface {
	Forward(predicted, actual *mat.Dense) float64
	Backward(predicted, actual *mat.Dense) []float64
}

func NewLossFunction(name string) lossFunction {
	switch name {
	case "square":
		return &SquareLoss{}
	case "crossentropy":
		return &CrossEntropyLoss{}
	default:
		// TODO: should we panic, maybe?
		return &UnknownLoss{}
	}
}

type SquareLoss struct{}

func (l *SquareLoss) Forward(predicted, actual *mat.Dense) float64 {
	rows, _ := predicted.Dims()
	error := 0.0
	for i := 0; i < rows; i++ {
		error += math.Pow(predicted.At(i, 0)-actual.At(i, 0), 2)
	}

	return 1.0 / float64(rows) * error
}

func (l *SquareLoss) Backward(predicted, actual *mat.Dense) []float64 {
	// TODO: panic when having more columns then 1 and not in same shape
	predictedVector := ArrayFromMatrix(predicted)
	actualVector := ArrayFromMatrix(actual)
	rows := len(predictedVector)

	errors := make([]float64, rows)
	for i := 0; i < rows; i++ {
		errors[i] = 2 * (predictedVector[i] - actualVector[i]) / float64(rows)
	}

	return errors
}

type CrossEntropyLoss struct{}

func (l *CrossEntropyLoss) Forward(predicted, actual *mat.Dense) float64 {
	rows, _ := predicted.Dims()
	error := 0.0
	for i := 0; i < rows; i++ {
		error += actual.At(i, 0) * math.Log(predicted.At(i, 0))
	}

	return -1.0 / float64(rows) * error
}

func (l *CrossEntropyLoss) Backward(predicted, actual *mat.Dense) []float64 {
	epsilon := 0.000001

	// TODO: panic when having more columns then 1 and not in same shape
	predictedVector := ArrayFromMatrix(predicted)
	actualVector := ArrayFromMatrix(actual)
	rows := len(predictedVector)

	errors := make([]float64, rows)
	for i := 0; i < rows; i++ {
		errors[i] = -(actualVector[i] / (predictedVector[i] + epsilon))
	}

	return errors
}

type UnknownLoss struct{}

func (l *UnknownLoss) Forward(predicted, actual *mat.Dense) float64 {
	return 0.0
}

func (l *UnknownLoss) Backward(predicted, actual *mat.Dense) []float64 {
	rows, _ := predicted.Dims()
	return ZeroArray(rows)
}
