package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx/internal"
	"runtime"
)

type NDArray struct {
	ctx    Context
	dim    Dimension
	dtype  Dtype
	handle internal.NDArrayHandle
	err    error
}

func release(a *NDArray) {
	if a != nil {
		internal.ReleaseNDArrayHandle(a.handle)
	}
}

// idiomatic finalizer
func (a *NDArray) Close() error {
	release(a)
	return nil
}

func (a *NDArray) Release() {
	release(a)
}

func (a NDArray) Err() error {
	return a.err
}

func Array(tp Dtype, d Dimension) *NDArray {
	return CPU.Array(tp, d)
}

func Errayf(s string, a ...interface{}) *NDArray {
	return &NDArray{
		err: fmt.Errorf(s, a...),
	}
}

func (c Context) Array(tp Dtype, d Dimension) *NDArray {
	if !d.Good() {
		return Errayf("failed to create array %v%v: bad dimension", tp.String(), d.String())
	}
	a := &NDArray{ctx: c, dim: d, dtype: tp}
	if h, e := internal.NewNDArrayHandle(c.DevType(), c.DevNo(), int(tp), d.Shape, d.Len); e != 0 {
		return Errayf("failed to create array %v%v: api error", tp.String(), d.String())
	} else {
		a.handle = h
		runtime.SetFinalizer(a, release)
	}
	return a
}

func (a *NDArray) Init(vals ...interface{}) *NDArray {
	return a
}

func (a *NDArray) Dtype() Dtype {
	return a.dtype
}

func (a *NDArray) Dim() Dimension {
	return a.dim
}

func (a *NDArray) Cast(dt Dtype) *NDArray {
	return nil
}

func (a *NDArray) Reshape(dim Dimension) *NDArray {
	return nil
}

func (a *NDArray) Data() []byte {
	return nil
}

type Variant struct {
	Value interface{}
}

func (v Variant) Float32() float32 {
	switch x := v.Value.(type) {
	case float32:
		return x
	case float64:
		return float32(x)
	case int64:
		return float32(x)
	case int32:
		return float32(x)
	case int8:
		return float32(x)
	case uint8:
		return float32(x)
	}
	return 0
}

func (a *NDArray) Get(idx ...int) Variant {
	return Variant{0}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func (a *NDArray) String() string {
	return ""
}

func (a *NDArray) Len(d int) int {
	if d < 0 || d >= 3 {
		return 0
	}
	if a.dim.Len <= d {
		return 1
	}
	return a.dim.Shape[d]
}

func (a *NDArray) Size() int {
	return a.dim.SizeOf(a.dtype)
}

func (a *NDArray) Values(dtype Dtype) (interface{}, error) {
	return nil, nil
}

func (a *NDArray) ValuesF32() []float32 {
	v, err := a.Values(Float32)
	if err != nil {
		panic(err.Error())
	}
	return v.([]float32)
}
