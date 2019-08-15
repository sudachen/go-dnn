package tests

import (
	"github.com/sudachen/go-dnn/mx"
	"gotest.tools/assert"
	"testing"
)

func Test_Forward(t *testing.T) {
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

	e := mx.Add(mx.Input, .3)

	f, err := mx.CPU.Bind(mx.Dim(3), e)
	assert.NilError(t, err)
	defer f.Release()

	for n := 0; n < len(input); n++ {
		f.Input.Init(input[0])
		err = f.Forward(false)
		assert.NilError(t, err)
		assert.Assert(t, func() bool {
			vals := f.Output.ValuesF32()
			for i := 0; i < 3; i++ {
				if vals[i] != output[0][i] {
					return false
				}
			}
			return true
		})
	}
}
