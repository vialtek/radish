package radish

import (
	"testing"
)

func vectorEquals(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func TestValidEncoding(t *testing.T) {
	labels := []string{"cat", "dog", "rabbit"}
	encoder := NewOneHotEncoder(labels)

	oneHotVector := encoder.Encode("cat")
	expectedVector := []float64{1, 0, 0}
	if !vectorEquals(oneHotVector, expectedVector) {
		t.Error("oneHotVector is not equal to expectedVector", oneHotVector, expectedVector)
	}
}

func TestNonUniqueLabels(t *testing.T) {
	labels := []string{"cat", "dog", "dog", "cat", "rabbit", "cat"}
	encoder := NewOneHotEncoder(labels)

	oneHotVector := encoder.Encode("rabbit")
	expectedVector := []float64{0, 0, 1}
	if !vectorEquals(oneHotVector, expectedVector) {
		t.Error("oneHotVector is not equal to expectedVector", oneHotVector, expectedVector)
	}
}

func TestEncodingWithUnknowLabel(t *testing.T) {
	labels := []string{"cat", "dog", "rabbit"}
	encoder := NewOneHotEncoder(labels)

	oneHotVector := encoder.Encode("fox")
	expectedVector := []float64{0, 0, 0}
	if !vectorEquals(oneHotVector, expectedVector) {
		t.Error("oneHotVector is not equal to expectedVector", oneHotVector, expectedVector)
	}
}

func TestEncodingListOfLabels(t *testing.T) {
	labels := []string{"cat", "dog", "rabbit"}
	encoder := NewOneHotEncoder(labels)

	labelsToEncode := []string{"cat", "dog", "dog"}
	encodedLabels := encoder.EncodeList(labelsToEncode)

	if len(encodedLabels) != len(labelsToEncode){
		t.Error("EncodeList list size do not match", encodedLabels, len(labelsToEncode))
	}

	if !vectorEquals(encodedLabels[0], encoder.Encode(labelsToEncode[0])){
		t.Error("Encoding does not match", encodedLabels[0], encoder.Encode(labelsToEncode[0]))
	}
}
