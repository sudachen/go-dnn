package capi

/*
#cgo CFLAGS: -I/opt/mxnet/include
#cgo LDFLAGS: -L/opt/mxnet/lib -lmxnet -Wl,-rpath=/opt/mxnet/lib
#include <mxnet/c_api.h>
#include <stdlib.h>

static int imperative_invoke1_inplace(AtomicSymbolCreator ent, NDArrayHandle out, int ano, const char **keys, const char **vals) {
	NDArrayHandle* out1[1] = {&out};
	int nout = 1;
	int err = MXImperativeInvoke(ent, 0, NULL, &nout, &out1[0], ano, keys, vals);
	return err;
}

static int imperative_invoke1_inout(AtomicSymbolCreator ent, NDArrayHandle in, NDArrayHandle out, int ano, const char **keys, const char **vals) {
	NDArrayHandle* out1[1] = {&out};
	int nout = 1;
	int err = MXImperativeInvoke(ent, 1, &in, &nout, &out1[0], ano, keys, vals);
	return err;
}

int NNSymbolListInputNames(SymbolHandle symbol,
                           int option,
                           unsigned int *out_size,
                           const char ***out_str_array);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var GpuCount int = 0
var LibVersion = 0
var mxkeys [KeyNoKey]*C.char
var mxentry [OpNoOp]C.AtomicSymbolCreator
var binopkeys [2]*C.char
var scalarkeys [1]*C.char

type NDArrayHandle = C.NDArrayHandle
type SymbolHandle = C.SymbolHandle
type ExecutorHandle = C.ExecutorHandle

func ReleaseNDArry(handle NDArrayHandle) {
	if handle != nil {
		C.MXNDArrayFree(handle)
	}
}

func ReleaseSymbol(handle SymbolHandle) {
	if handle != nil {
		C.MXSymbolFree(handle)
	}
}

func ReleaseExecutor(handle ExecutorHandle) {
	if handle != nil {
		C.MXExecutorFree(handle)
	}
}

func mxLastError() string {
	e := C.MXGetLastError()
	return C.GoString(e)
}

func init() {

	var v C.int
	C.MXGetVersion(&v)
	LibVersion = int(v)

	var c C.int
	C.MXGetGPUCount(&c)
	GpuCount = int(c)

	for i := KeyEmpty + 1; i < KeyNoKey; i++ {
		mxkeys[i] = C.CString(i.Value())
	}

	binopkeys[0] = mxkeys[KeyLhs]
	binopkeys[1] = mxkeys[KeyRhs]
	scalarkeys[0] = mxkeys[KeyData]

	var ascv *C.AtomicSymbolCreator
	var ascn C.uint

	if e := C.MXSymbolListAtomicSymbolCreators(&ascn, &ascv); e != 0 {
		panic("failed to gather symbols from mxnet")
	}

	m := map[string]MxnetOp{}
	for op := OpEmpty + 1; op < OpNoOp; op++ {
		m[op.Value()] = op
	}

	for i := uintptr(0); i < uintptr(ascn); i++ {
		a := *(*C.AtomicSymbolCreator)(unsafe.Pointer(uintptr(unsafe.Pointer(ascv)) + i*unsafe.Sizeof(*ascv)))
		var n *C.char
		if e := C.MXSymbolGetAtomicSymbolName(a, &n); e != 0 {
			panic(fmt.Sprintf("failed to gather name for symbol %x", a))
		}
		//fmt.Println(C.GoString(n))
		if ent, ok := m[C.GoString(n)]; ok {
			mxentry[ent] = a
		}
	}
}

const maxArgsCount = 16

func fillargs(keys []*C.char, vals []*C.char, ap []interface{}) int {
	i := 0
	for len(ap) != 0 && i < maxArgsCount*2 {
		keys[i] = mxkeys[ap[0].(MxnetKey)]
		vals[i] = C.CString(fmt.Sprint(ap[1]))
		i++
		ap = ap[2:]
	}
	return i
}

func ImperativeInvokeInplace1(op MxnetOp, h NDArrayHandle, a ...interface{}) error {
	if h == nil {
		return fmt.Errorf("uninitialized or broken array")
	}

	var keys [maxArgsCount]*C.char
	var vals [maxArgsCount]*C.char
	ano := C.int(fillargs(keys[:], vals[:], a))
	defer func() {
		for _, v := range vals {
			if v != nil {
				C.free(unsafe.Pointer(v))
			}
		}
	}()

	if ent := mxentry[op]; ent != nil {
		if e := C.imperative_invoke1_inplace(ent, C.NDArrayHandle(h), ano, &keys[0], &vals[0]); e != 0 {
			return fmt.Errorf("maxnet api error: %v", op.Value())
		}
	} else {
		return fmt.Errorf("unresolved API entry %v", op.Value())
	}
	return nil
}

func ImperativeInvokeInOut1(op MxnetOp, h NDArrayHandle, o NDArrayHandle, a ...interface{}) error {
	if h == nil {
		return fmt.Errorf("uninitialized or broken input array")
	}
	if o == nil {
		return fmt.Errorf("uninitialized or broken output array")
	}

	var keys [maxArgsCount]*C.char
	var vals [maxArgsCount]*C.char
	ano := C.int(fillargs(keys[:], vals[:], a))
	defer func() {
		for _, v := range vals {
			if v != nil {
				C.free(unsafe.Pointer(v))
			}
		}
	}()
	if ent := mxentry[op]; ent != nil {
		if e := C.imperative_invoke1_inout(ent, C.NDArrayHandle(h), C.NDArrayHandle(o), ano, &keys[0], &vals[0]); e != 0 {
			return fmt.Errorf("maxnet api error: %v", op.Value())
		}
	} else {
		return fmt.Errorf("unresolved API entry %v", op.Value())
	}
	return nil
}

func NewNDArrayHandle(devType int, devNo int, dtype int, shape [4]int, slen int) (NDArrayHandle, int) {
	var a C.NDArrayHandle
	s := [4]C.uint{C.uint(shape[0]), C.uint(shape[1]), C.uint(shape[2]), C.uint(shape[3])}
	e := C.MXNDArrayCreateEx(&s[0], C.uint(slen), C.int(devType), C.int(devNo), 0, C.int(dtype), &a)
	return NDArrayHandle(a), int(e)
}

func GetNDArrayRawData(handle NDArrayHandle, p unsafe.Pointer, len int) int {
	if handle != nil {
		e := C.MXNDArraySyncCopyToCPU(handle, p, C.ulong(len))
		return int(e)
	}
	return -1
}

func SetNDArrayRawData(handle NDArrayHandle, p unsafe.Pointer, len int) int {
	if handle != nil {
		e := C.MXNDArraySyncCopyFromCPU(handle, p, C.ulong(len))
		return int(e)
	}
	return -1
}

func CreateVariable(name string) (SymbolHandle, int) {
	var r SymbolHandle
	str := C.CString(name)
	defer C.free(unsafe.Pointer(str))
	e := C.MXSymbolCreateVariable(str, &r)
	return r, int(e)
}

func NewSymbol(op MxnetOp, attr map[MxnetKey]string, a ...interface{}) (SymbolHandle, error) {

	var keys [maxArgsCount]*C.char
	var vals [maxArgsCount]*C.char

	if len(a)+len(attr) >= maxArgsCount {
		return nil, fmt.Errorf("number of keys and vals must be less than %v", maxArgsCount)
	}

	i := fillargs(keys[:], vals[:], a)
	if attr != nil {
		for k, v := range attr {
			keys[i] = mxkeys[k]
			vals[i] = C.CString(v)
			i++
		}
	}

	defer func() {
		for _, v := range vals[:i] {
			C.free(unsafe.Pointer(v))
		}
	}()

	ano := C.int(len(a)/2 + len(attr))

	var h SymbolHandle
	if ent := mxentry[op]; ent != nil {
		if e := C.MXSymbolCreateAtomicSymbol(ent, C.uint(ano), &keys[0], &vals[0], &h); e != 0 {
			return nil, fmt.Errorf("failed to create mxnet symbol %v: %v", op.Value(), mxLastError())
		}
	} else {
		return nil, fmt.Errorf("unresolved API entry %v", op.Value())
	}

	return h, nil
}

func ComposeSymbol(handle SymbolHandle, name string, a ...SymbolHandle) error {
	str := C.CString(name)
	defer C.free(unsafe.Pointer(str))
	//keys := binopkeys[:]
	//if len(a) < 2 { keys = scalarkeys[:] }
	if e := C.MXSymbolCompose(handle, str, C.uint(len(a)), nil, &a[0]); e != 0 {
		return fmt.Errorf("failed to compose mxnet symbol %v: %v", name, mxLastError())
	}
	return nil
}

func ListArguments(handle SymbolHandle) ([]string, error) {
	var (
		i      int
		e      C.int
		out_nn C.uint
		out_ns **C.char
		r      []string
	)

	e = C.MXSymbolListArguments(
		handle,
		&out_nn,
		&out_ns)

	if e != 0 {
		return nil, fmt.Errorf("failed to request input names from mxnet: %v", mxLastError())
	}

	name_at := func(i int) string {
		p := (uintptr)(unsafe.Pointer(out_ns)) + uintptr(i)*unsafe.Sizeof(uintptr(0))
		return C.GoString(*(**C.char)(unsafe.Pointer(p)))
	}

	r = make([]string, int(out_nn))

	for i = 0; i < int(out_nn); i++ {
		r[i] = name_at(i)
	}

	return r, nil
}

func InferShapes(handle SymbolHandle, with map[string][]int) (map[string][]int, error) {

	if len(with) > maxArgsCount {
		return nil, fmt.Errorf("to many shapes in args")
	}

	var (
		keys                  [maxArgsCount]*C.char
		si                    [maxArgsCount]C.uint
		sd                    [maxArgsCount * 4]C.uint
		in_ss, out_ss, aux_ss C.uint
		in_sn, out_sn, aux_sn *C.uint
		in_sd, out_sd, aux_sd **C.uint
		complete              C.int
	)

	i, j := 0, 0
	for s, v := range with {
		keys[i] = C.CString(s)
		i++
		for _, t := range v {
			sd[j] = C.uint(t)
			j++
		}
		si[i] = C.uint(j)
	}

	defer func() {
		for _, p := range keys {
			C.free(unsafe.Pointer(p))
		}
	}()

	e := C.MXSymbolInferShape(handle,
		C.uint(len(with)),
		&keys[0],
		&si[0],
		&sd[0],
		&in_ss, &in_sn, &in_sd,
		&out_ss, &out_sn, &out_sd,
		&aux_ss, &aux_sn, &aux_sd,
		&complete)
	if e != 0 {
		return nil, fmt.Errorf("failed to request shapes from mxnet: %v", mxLastError())
	}

	shape_at := func(i int, d *C.uint, s **C.uint) []int {
		pd := uintptr(unsafe.Pointer(d)) + uintptr(i)*unsafe.Sizeof(C.uint(0))
		n := int(*(*C.uint)(unsafe.Pointer(pd)))
		r := make([]int, n)
		ps := *(**C.uint)(unsafe.Pointer(uintptr(unsafe.Pointer(s)) + uintptr(i)*unsafe.Sizeof(uintptr(0))))

		for j := 0; j < n; j++ {
			r[j] = int(*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(ps)) + uintptr(j)*unsafe.Sizeof(C.uint(0)))))
		}

		return r
	}

	r := make(map[string][]int)
	names, err := ListArguments(handle)
	if err != nil {
		return nil, err
	}

	for i, name := range names {
		if _, ok := with[name]; !ok {
			r[name] = shape_at(i, in_sn, in_sd)
		}
	}

	r["_output"] = shape_at(0, out_sn, out_sd)

	return r, nil
}

func GroupSymbols(s []SymbolHandle) (SymbolHandle, error) {
	var r SymbolHandle
	e := C.MXSymbolCreateGroup(C.uint(len(s)), &s[0], &r)
	if e != 0 {
		return nil, fmt.Errorf("failed to group mxnet symbols: %v", mxLastError())
	}
	return r, nil
}

func Bind(symbol SymbolHandle, devType, devNo int, args []NDArrayHandle, grads []NDArrayHandle) (ExecutorHandle, error) {
	var r ExecutorHandle

	ga := make([]C.uint, len(args))
	for i := range ga {
		ga[i] = 1
	}

	e := C.MXExecutorBind(
		symbol,
		C.int(devType),
		C.int(devNo),
		C.uint(len(args)),
		&args[0],
		&grads[0],
		&ga[0],
		C.uint(0),
		nil,
		&r)

	if e != 0 {
		return nil, fmt.Errorf("failed to bind mxnet symbols: %v", mxLastError())
	}

	return r, nil
}

func GetOutputs(exec ExecutorHandle) ([]NDArrayHandle, error) {
	var (
		n C.uint
		a *NDArrayHandle
		e C.int
	)
	if e = C.MXExecutorOutputs(exec, &n, &a); e != 0 {
		return nil, fmt.Errorf("failed get mxnet outputs: %v", mxLastError())
	}
	r := make([]NDArrayHandle, int(n))
	for i := range r {
		p := uintptr(unsafe.Pointer(a)) + uintptr(i)*unsafe.Sizeof((*NDArrayHandle)(nil))
		r[i] = *(*NDArrayHandle)(unsafe.Pointer(p))
	}
	return r, nil
}

func Forward(exec ExecutorHandle, isTrain bool) error {
	t := C.int(0)
	if isTrain {
		t = C.int(1)
	}
	if e := C.MXExecutorForward(exec, t); e != 0 {
		return fmt.Errorf("failed on mxnet forward: %v", mxLastError())
	}
	return nil
}
