package tests

import (
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"gotest.tools/assert"
	"testing"
)

/*
func nn_Test() {
	nn.Connect(
		&nn.Convolution{
			Kernel: mx.Dim(3,3),
			Filters: 25,
		},
		&nn.FullConnected{
			Count: 100,
		})
}
*/

func Test_Lambda_Forward(t *testing.T) {
	var err error

	input := [][]float32{
		{1, 2, 3},
		{1, 1, 1},
		{0, 0, 0},
	}

	output := [][]float32{
		{3, 6, 9},
		{3, 3, 3},
		{0, 0, 0},
	}

	nb := &nn.Lambda{func(input *mx.Symbol) *mx.Symbol { return mx.Add(input, .3) }}
	net, err := nn.Bind(mx.CPU, nb, mx.Dim(1, 3))
	assert.NilError(t, err)
	assert.Assert(t, net != nil)
	defer net.Release()

	for n := 0; n < len(input); n++ {
		err = net.Graph.Input.SetValues(input[0])
		assert.NilError(t, err)
		err = net.Predict()
		assert.NilError(t, err)
		assert.Check(t, func() bool {
			vals := net.Graph.Output.ValuesF32()
			for i := 0; i < 3; i++ {
				if vals[i] != output[0][i] {
					return false
				}
			}
			return true
		}())
	}
}
