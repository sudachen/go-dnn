package tests

import (
	"github.com/sudachen/go-dnn/mx"
	"gotest.tools/assert"
	"reflect"
	"testing"
)

func f_Array1(ctx mx.Context, t *testing.T) {
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
	f_Array1(mx.CPU, t)
	for gno := 0; gno < mx.GpuCount(); gno++ {
		f_Array1(mx.Gpu(gno), t)
	}
}

var dtypeMap = map[mx.Dtype]reflect.Type{
	mx.Float64: reflect.TypeOf(float64(0)),
	mx.Float32: reflect.TypeOf(float32(0)),
	mx.Float16: reflect.TypeOf(float32(0)),
	mx.Int8:    reflect.TypeOf(int8(0)),
	mx.Uint8:   reflect.TypeOf(uint8(0)),
	mx.Int32:   reflect.TypeOf(int32(0)),
	mx.Int64:   reflect.TypeOf(int64(0)),
}

type array2_ds_t struct {
	mx.Dtype
	vals interface{}
}

var array2_ds = []array2_ds_t{
	array2_ds_t{mx.Float32, []interface{}{.1, int(2), int64(3), float64(4), .0005}},
	array2_ds_t{mx.Int32, []interface{}{.1, int(2), int64(3), float64(4), .0005}},
	array2_ds_t{mx.Uint8, []int{1, 2, 3, 4, 5}},
	// Float16 has a round error so using integer values only
	array2_ds_t{mx.Float16, []interface{}{1, int(2), int64(3), float64(4), 5}},
}

func compare(t *testing.T, data, result interface{}, dt mx.Dtype, no int) bool {
	v0 := reflect.ValueOf(data)
	if v0.Kind() != reflect.Slice && v0.Kind() != reflect.Array {
		t.Errorf("test data is not slice")
		return false
	}
	v1 := reflect.ValueOf(result)
	if v1.Kind() != reflect.Slice && v1.Kind() != reflect.Array {
		t.Errorf("test result is not slice")
		return false
	}
	if v0.Len() > v1.Len() {
		t.Errorf("test data is longer than test result")
		return false
	}
	for i := 0; i < v0.Len(); i++ {
		q := func(v reflect.Value) reflect.Value {
			if v.Kind() == reflect.Interface {
				v = v.Elem()
			}
			return v.Convert(dtypeMap[dt])
		}
		val0 := q(v0.Index(i))
		val1 := q(v1.Index(i))
		if !reflect.DeepEqual(val0.Interface(), val1.Interface()) {
			t.Errorf("%v at %v: %v != %v, %v", no, i, val0, val1, dt)
			return false
		}
	}
	return true
}

func f_Array2(ctx mx.Context, t *testing.T) {
	t.Logf("Array on %v", ctx)
	for no, v := range array2_ds {
		a := ctx.Array(v.Dtype, mx.Dim(5)).Init(v.vals)
		defer a.Release()
		if v.Dtype == mx.Float16 && ctx.IsGPU() && a.Err() != nil {
			t.Logf("Float16 is unsupported on GPU, skipped")
			continue // can be unsupported on GPU
		}
		assert.NilError(t, a.Err())
		dt := v.Dtype
		if dt == mx.Float16 {
			dt = mx.Float32
		}
		vals, err := a.Values(dt)
		assert.NilError(t, err)
		assert.Check(t, compare(t, v.vals, vals, dt, no))
	}
}

func Test_Array2(t *testing.T) {
	f_Array2(mx.CPU, t)
	for gno := 0; gno < mx.GpuCount(); gno++ {
		f_Array2(mx.Gpu(gno), t)
	}
}

func Test_Array3(t *testing.T) {
	var err error
	ds := []int{1, 2, 3, 4, 5}
	a := mx.CPU.Array(mx.Float16, mx.Dim(5)).Init(1, 2, 3, 4, 5)
	defer a.Release()
	assert.NilError(t, a.Err())
	_, err = a.Values(mx.Float16)
	assert.ErrorContains(t, err, "Float16")
	v, err := a.Values(mx.Float32)
	assert.NilError(t, err)
	assert.Check(t, compare(t, ds, v, mx.Float32, 0))
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