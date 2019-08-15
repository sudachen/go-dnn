package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx/internal"
	"reflect"
	"runtime"
	"unsafe"
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

func (a *NDArray) erray(err error) *NDArray {
	a.err = err
	return a
}

func (a *NDArray) errayf(s string, v ...interface{}) *NDArray {
	a.err = fmt.Errorf(s, v...)
	return a
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

func (c Context) CopyAs(a *NDArray, dtype Dtype) *NDArray {
	if a.handle == nil {
		return Errayf("can't copy broken array")
	}
	b := c.Array(dtype, a.dim)
	if err := internal.ImperativeInvokeInOut1(internal.OpCopyTo, a.handle, b.handle); err != nil {
		b.Release()
		return &NDArray{err: err}
	}
	return b
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

var typemap = map[Dtype]reflect.Type{
	Float64: reflect.TypeOf(float64(0)),
	Float32: reflect.TypeOf(float32(0)),
	Int8:    reflect.TypeOf(int8(0)),
	Uint8:   reflect.TypeOf(uint8(0)),
	Int32:   reflect.TypeOf(int32(0)),
	Int64:   reflect.TypeOf(int64(0)),
}

func copyTo(s reflect.Value, n int, v0 reflect.Value, dt reflect.Type) (int, error) {
	var err error
	if v0.Kind() == reflect.Interface {
		v0 = v0.Elem()
	}
	if v0.Kind() == reflect.Slice || v0.Kind() == reflect.Array {
		for i := 0; i < v0.Len(); i++ {
			n, err = copyTo(s, n, v0.Index(i), dt)
			if err != nil {
				return 0, err
			}
		}
	} else {
		switch v0.Kind() {
		case reflect.Int, reflect.Int8, reflect.Uint8, reflect.Int16, reflect.Uint16,
			reflect.Int32, reflect.Uint32, reflect.Int64, reflect.Uint64,
			reflect.Float32, reflect.Float64:
			s.Index(n).Set(v0.Convert(dt))
			n++
		default:
			return 0, fmt.Errorf("can't initialize with non numeric type %v", v0.Type())
		}
	}
	return n, nil
}

func (a *NDArray) Init(vals ...interface{}) *NDArray {
	if a.handle == nil {
		return Errayf("can't initialize broken array")
	}

	if a.dtype == Float16 {
		q := CPU.CopyAs(a, Float32).Init(vals...)
		defer q.Release()
		if err := q.Err(); err != nil {
			return a.erray(err)
		}
		if err := internal.ImperativeInvokeInOut1(internal.OpCopyTo, q.handle, a.handle); err != nil {
			return a.errayf("failed copy temporal Float32 array to Float16 target")
		}
		return a
	}

	dt, ok := typemap[a.dtype]
	if !ok {
		return a.errayf("initialization with dtype %v is unsupportd", a.dtype)
	}
	s := reflect.MakeSlice(reflect.SliceOf(dt), a.dim.Total(), a.dim.Total())
	if _, err := copyTo(s, 0, reflect.ValueOf(vals), dt); err != nil {
		return a.erray(err)
	}
	e := internal.SetNDArrayRawData(a.handle, unsafe.Pointer(s.Index(0).UnsafeAddr()), a.dim.Total())
	if e != 0 {
		return a.errayf("failed to initialize array with raw data")
	}
	return a
}

func (a *NDArray) Raw() []byte {
	ln := a.dim.Total()
	bs := make([]byte, ln)
	internal.GetNDArrayRawData(a.handle, unsafe.Pointer(&bs[0]), ln)
	return bs
}

func (a *NDArray) Values(dtype Dtype) (interface{}, error) {
	if dtype == Float16 {
		return nil, fmt.Errorf("can't gate values in Float16 format")
	}
	q := a
	ln := q.dim.Total()
	if q.dtype != dtype {
		q = CPU.CopyAs(q, dtype)
		defer q.Release()
	}
	vals := reflect.MakeSlice(reflect.SliceOf(typemap[dtype]), ln, ln)
	if e := internal.GetNDArrayRawData(q.handle, unsafe.Pointer(vals.Index(0).UnsafeAddr()), ln); e != 0 {
		return nil, fmt.Errorf("failed to copy raw data")
	}
	return vals.Interface(), nil
}

func (a *NDArray) ValuesF32() []float32 {
	v, err := a.Values(Float32)
	if err != nil {
		panic(err.Error())
	}
	return v.([]float32)
}