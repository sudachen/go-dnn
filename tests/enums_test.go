package tests

import (
	"github.com/sudachen/go-dnn/mx"
	"gotest.tools/assert"
	"strings"
	"testing"
)

func Test_Context(t *testing.T) {
	var c mx.Context
	a := c.Array(mx.Float32, mx.Dim(1))
	assert.Assert(t, a.Err() != nil)
	assert.Assert(t, c.String() == "NullContext")
	assert.Assert(t, mx.Context(9999999).String() == "InvalidContext")
	c = mx.GPU0
	assert.Assert(t, c == mx.NullContext || strings.Index(c.String(), "GPU") == 0)
	c = mx.Gpu(0)
	assert.Assert(t, c == mx.NullContext || strings.Index(c.String(), "GPU") == 0)
	assert.Assert(t, c == mx.GPU0 || c == mx.NullContext)
	c = mx.Gpu(1)
	assert.Assert(t, c == mx.GPU1 || c == mx.NullContext)
}

func Test_Dtype(t *testing.T) {
	a := []mx.Dtype{mx.Uint8, mx.Int8, mx.Int32, mx.Int64, mx.Float16, mx.Float32, mx.Float64}
	s := []string{"Uint8", "Int8", "Int32", "Int64", "Float16", "Float32", "Float64"}
	q := []int{1, 1, 4, 8, 2, 4, 8}
	for n, v := range a {
		assert.Assert(t, v.String() == s[n])
		assert.Assert(t, v.Size() == q[n])
		assertPanic(t, func() { _ = mx.Dtype(100001).String() })
		assertPanic(t, func() { _ = mx.Dtype(100001).Size() })
	}
}

