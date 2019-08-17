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

	nb := &nn.Lambda{func(input *mx.Symbol) *mx.Symbol { return mx.Ns("ns", mx.Mul(input, 3)) }}
	net, err := nn.Bind(mx.CPU, nb, mx.Dim(1, 3), nn.L1Loss)
	assert.NilError(t, err)
	assert.Assert(t, net != nil)
	defer net.Release()

	assert.Assert(t, net.Graph.Output.Depth() == 2)
	assert.Assert(t, net.Graph.Output.Len(0) == 1)
	assert.Assert(t, net.Graph.Output.Len(1) == 3)

	for n := 0; n < len(input); n++ {
		assert.NilError(t, err)
		r, err := net.Predict(input[n : n+1])
		assert.NilError(t, err)
		assert.Check(t, func() bool {
			for i := 0; i < 3; i++ {
				if r[0][i] != output[n][i] {
					t.Errorf("%v != %v", r[0], output[n])
					return false
				}
			}
			return true
		}())
	}
}

func Test_nn1(t *testing.T) {
}
