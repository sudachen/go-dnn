package tests

import (
	"github.com/sudachen/go-dnn/mx"
	"gotest.tools/assert"
	"testing"
)

func f_Array(ctx mx.Context, t *testing.T) {
	t.Logf("Array on %v", ctx)
	a := ctx.Array(mx.Int32, mx.Dim(100))
	defer a.Release()
	assert.NilError(t, a.Err())
	assert.Assert(t, a != nil)
	assert.Assert(t, a.Dtype().String() == "Int32")
	assert.Assert(t, a.Dim().String() == "(100)")
	b := mx.Array(mx.Float32, mx.Dim(100, 10))
	defer b.Release()
	assert.NilError(t, b.Err())
	assert.Assert(t, b != nil)
	assert.Assert(t, b.Dtype().String() == "Float32")
	assert.Assert(t, b.Dim().String() == "(100,10)")
	c := mx.Array(mx.Uint8, mx.Dim(100, 10, 1))
	defer c.Release()
	assert.NilError(t, c.Err())
	assert.Assert(t, c.Dtype().String() == "Uint8")
	assert.Assert(t, c.Dim().String() == "(100,10,1)")
	d := mx.Array(mx.Int64, mx.Dim(100000000000, 100000000, 1000000000))
	assert.ErrorContains(t, d.Err(), "failed to create array")
	d = mx.Array(mx.Int64, mx.Dim())
	assert.ErrorContains(t, d.Err(), "bad dimension")
	d = mx.Array(mx.Int64, mx.Dim(-1, 3))
	assert.ErrorContains(t, d.Err(), "bad dimension")
	d = mx.Array(mx.Int64, mx.Dim(1, 3, 10, 100, 2))
	assert.ErrorContains(t, d.Err(), "bad dimension")
}

func Test_Array1(t *testing.T) {
	f_Array(mx.CPU, t)
	for gno := 0; gno < mx.GpuCount(); gno++ {
		f_Array(mx.Gpu(gno), t)
	}
}

func f_Random(ctx mx.Context, t *testing.T) {
	t.Logf("Random_Uniform on %v", ctx)
	a := ctx.Array(mx.Float32, mx.Dim(1, 3)).Uniform(0, 1)
	defer a.Release()
	assert.NilError(t, a.Err())
	t.Log(a.ValuesF32())
}

func Test_Random(t *testing.T) {
	f_Random(mx.CPU, t)
	if mx.GpuCount() > 0 {
		f_Random(mx.GPU0, t)
	}
}
