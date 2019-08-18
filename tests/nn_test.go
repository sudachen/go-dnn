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
	net, err := nn.Bind(mx.CPU, nb, mx.Dim(1, 3), nil)
	assert.NilError(t, err)
	assert.Assert(t, net != nil)
	defer net.Release()

	assert.Assert(t, net.Graph.Outputs[0].Depth() == 2)
	assert.Assert(t, net.Graph.Outputs[0].Len(0) == 1)
	assert.Assert(t, net.Graph.Outputs[0].Len(1) == 3)

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

func f_nn1(t *testing.T, opt nn.OptimizerConf) {

	f := func(x *mx.Symbol) *mx.Symbol {
		return mx.Pow(mx.Add(x, mx.Var("_offset", mx.Autograd)), 2)
	}

	net, err := nn.Bind(mx.CPU, &nn.Lambda{f}, mx.Dim(1, 2), opt)
	assert.NilError(t, err)
	assert.Assert(t, net != nil)
	defer net.Release()

	input := []float32{3, 3}
	label := []float32{0, 0}

	assert.Assert(t, net.Graph.Outputs[0].Depth() == 2)
	assert.Assert(t, net.Graph.Outputs[0].Len(0) == 1)
	assert.Assert(t, net.Graph.Outputs[0].Len(1) == 2)

	for n := 0; n < 100; n++ {
		assert.NilError(t, err)
		err := net.Train(input, label)
		assert.NilError(t, err)
		v := net.Graph.Params["_offset"].Data.ValuesF32()
		if v[0]+input[0] < 0.1 && v[1]+input[1] < 0.1  {
			t.Logf("%d %v %v %v %v", n,
				net.Graph.Loss.ValuesF32(),
				net.Graph.Outputs[0].ValuesF32(),
				net.Graph.Params["_offset"].Data.ValuesF32(),
				net.Graph.Params["_offset"].Grad.ValuesF32())
			break
		}
	}

	v := net.Graph.Params["_offset"].Data.ValuesF32()
	assert.Assert(t, v[0]+input[0] < 0.1 && v[1]+input[1] < 0.1)
}

func Test_nn1(t *testing.T) {
	f_nn1(t,&nn.SGD{Lr:.01})
	f_nn1(t,&nn.Adam{Lr:.1})
	f_nn1(t,&nn.Adam{Lr:.1,Loss:nn.L0Loss})
}
