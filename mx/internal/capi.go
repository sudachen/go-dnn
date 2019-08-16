package internal

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

type NDArrayHandle C.NDArrayHandle
type SymbolHandle C.SymbolHandle

func ReleaseNDArry(handle NDArrayHandle) {
	if handle != nil {
		C.MXNDArrayFree(C.NDArrayHandle(handle))
	}
}

func ReleaseSymbol(handle SymbolHandle) {
	if handle != nil {
		C.MXSymbolFree(C.SymbolHandle(handle))
	}
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
		e := C.MXNDArraySyncCopyToCPU(C.NDArrayHandle(handle), p, C.ulong(len))
		return int(e)
	}
	return -1
}

func SetNDArrayRawData(handle NDArrayHandle, p unsafe.Pointer, len int) int {
	if handle != nil {
		e := C.MXNDArraySyncCopyFromCPU(C.NDArrayHandle(handle), p, C.ulong(len))
		return int(e)
	}
	return -1
}

func CreateVariable(name string) (SymbolHandle, int) {
	var r C.SymbolHandle
	str := C.CString(name)
	defer C.free(unsafe.Pointer(str))
	e := C.MXSymbolCreateVariable(str, &r)
	return SymbolHandle(r), int(e)
}

func NewSymbol(op MxnetOp, attr map[MxnetKey]string, a ...interface{}) (SymbolHandle, error) {

	var keys [maxArgsCount]*C.char
	var vals [maxArgsCount]*C.char

	if len(a)+len(attr) >= maxArgsCount {
		return nil, fmt.Errorf("number of keys and vals must be less than %v", maxArgsCount)
	}

	i := fillargs(keys[:], vals[:], a)
	for k, v := range attr {
		keys[i] = mxkeys[k]
		vals[i] = C.CString(fmt.Sprint(v))
		i++
	}

	defer func() {
		for _, v := range vals[:i] {
			C.free(unsafe.Pointer(v))
		}
	}()

	ano := C.int(len(a)/2 + len(attr))

	var h C.SymbolHandle
	if ent := mxentry[op]; ent != nil {
		if e := C.MXSymbolCreateAtomicSymbol(ent, C.uint(ano), &keys[0], &vals[0], &h); e != 0 {
			return nil, fmt.Errorf("failed to create mxnet symbol %v", op.Value())
		}
	} else {
		return nil, fmt.Errorf("unresolved API entry %v", op.Value())
	}

	return SymbolHandle(h), nil
}

func ComposeSymbol(handle SymbolHandle, name string, a ...SymbolHandle) error {
	str := C.CString(name)
	defer C.free(unsafe.Pointer(str))
	//keys := binopkeys[:]
	//if len(a) < 2 { keys = scalarkeys[:] }
	if e := C.MXSymbolCompose(C.SymbolHandle(handle), str, C.uint(len(a)), nil, (*C.SymbolHandle)(unsafe.Pointer(&a[0]))); e != 0 {
		return fmt.Errorf("failed to compose mxnet symbol %v", name)
	}
	return nil
}

func InferShapes(handle SymbolHandle, with map[string][]int) (map[string][]int, error) {
	return nil, nil
}
