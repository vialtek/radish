package radish

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type DenseLayer struct {
	activation string
	Weights    *mat.Dense
}

func NewDenseLayer(inputs, outputs int, activation string) *DenseLayer {
	rand.Seed(time.Now().UnixNano())
	randData := make([]float64, (inputs+1)*outputs)
	for i := range randData {
		randData[i] = rand.NormFloat64()
	}

	return &DenseLayer{
		activation: activation,
		Weights:    mat.NewDense(inputs+1, outputs, randData),
	}
}

func (l *DenseLayer) ForwardProp(input *mat.Dense) *mat.Dense {
	var fullInput, output, activatedOutput mat.Dense

	bias := mat.NewDense(1, 1, []float64{1})
	fullInput.Augment(input, bias)
	output.Mul(&fullInput, l.Weights)
	//activatedOutput.Apply(l.activationElem, &output)

	return &activatedOutput
}