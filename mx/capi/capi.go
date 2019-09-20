package capi

/*
#cgo CFLAGS: -I/opt/mxnet/include
#cgo LDFLAGS: -L/opt/mxnet/lib -lmxnet -Wl,-rpath=/opt/mxnet/lib
#include <mxnet/c_api.h>
#include <stdlib.h>
#include <string.h>

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

static int imperative_invokeN_inout(
	AtomicSymbolCreator ent,
	NDArrayHandle out,
	int ano, const char **keys, const char **vals,
	NDArrayHandle in0, NDArrayHandle in1,
	NDArrayHandle in2, NDArrayHandle in3)
{
	int nin = 0;
	NDArrayHandle* out1[1] = {&out};
	NDArrayHandle inN[4] = {in0,in1,in2,in3};
	for ( ;nin<4 && inN[nin]; ++nin) {}
	int nout = 1;
	int err = MXImperativeInvoke(ent, nin, inN, &nout, &out1[0], ano, keys, vals);
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
	"github.com/google/logger"
	"github.com/sudachen/go-dnn/fu"
	"unsafe"
)

var GpuCount int = 0
var LibVersion = 0
var mxkeys = map[MxnetKey]*C.char{}
var mxentry = map[MxnetOp]C.AtomicSymbolCreator{}

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

	var ascv *C.AtomicSymbolCreator
	var ascn C.uint

	if e := C.MXSymbolListAtomicSymbolCreators(&ascn, &ascv); e != 0 {
		panic("failed to gather symbols from mxnet")
	}

	m := map[string]MxnetOp{}
	for op := OpEmpty + 1; op < OpNoOp; op++ {
		m[op.Value()] = op
	}

	for i := 0; i < int(ascn); i++ {
		a := *(*C.AtomicSymbolCreator)(fu.Index(i, ascv))
		var n *C.char
		if e := C.MXSymbolGetAtomicSymbolName(a, &n); e != 0 {
			panic(fmt.Sprintf("failed to gather name for symbol %x", a))
		}
		if ent, ok := m[C.GoString(n)]; ok {
			mxentry[ent] = a
		}
	}

	notInit := false
	for n := range opmap {
		if _, ok := mxentry[n]; !ok {
			notInit = true
			logger.Errorf("mxnet operator %v is not found in shared library", n.Value())
		}
	}

	if notInit {
		logger.Infof("available operators:")
		for i := 0; i < int(ascn); i++ {
			a := *(*C.AtomicSymbolCreator)(fu.Index(i, ascv))
			var n *C.char
			if e := C.MXSymbolGetAtomicSymbolName(a, &n); e != 0 {
				panic(fmt.Sprintf("failed to gather name for symbol %x", a))
			}
			logger.Info(C.GoString(n))
		}
		panic("not initialized")
	}
}

func ImperativeInvokeInplace1(op MxnetOp, h NDArrayHandle, a ...interface{}) error {
	if h == nil {
		return fmt.Errorf("uninitialized or broken array")
	}

	var keys [MaxArgsCount]*C.char
	var vals [MaxArgsCount]*C.char
	ano := C.int(Fillargs(keys[:], vals[:], a))

	if ent := mxentry[op]; ent != nil {
		if e := C.imperative_invoke1_inplace(ent, C.NDArrayHandle(h), ano, &keys[0], &vals[0]); e != 0 {
			return fmt.Errorf("maxnet %v error: %v", op.Value(), mxLastError())
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

	var keys [MaxArgsCount]*C.char
	var vals [MaxArgsCount]*C.char
	ano := C.int(Fillargs(keys[:], vals[:], a))
	if ent := mxentry[op]; ent != nil {
		if e := C.imperative_invoke1_inout(ent, C.NDArrayHandle(h), C.NDArrayHandle(o), ano, &keys[0], &vals[0]); e != 0 {
			return fmt.Errorf("maxnet %v error: %v", op.Value(), mxLastError())
		}
	} else {
		return fmt.Errorf("unresolved API entry %v", op.Value())
	}
	return nil
}

func NewNDArrayHandle(devType int, devNo int, dtype int, shape [4]int, slen int) (NDArrayHandle, error) {
	var a C.NDArrayHandle
	s := [4]C.uint{C.uint(shape[0]), C.uint(shape[1]), C.uint(shape[2]), C.uint(shape[3])}
	if e := C.MXNDArrayCreateEx(&s[0], C.uint(slen), C.int(devType), C.int(devNo), 0, C.int(dtype), &a); e != 0 {
		return nil, fmt.Errorf("failed to create ndarry: %v", mxLastError())
	}
	return NDArrayHandle(a), nil
}

func GetNDArrayRawData(handle NDArrayHandle, p unsafe.Pointer, len int) error {
	if handle != nil {
		if e := C.MXNDArraySyncCopyToCPU(handle, p, C.ulong(len)); e != 0 {
			return fmt.Errorf("failed to get raw data: %v", mxLastError())
		}
	}
	return nil
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

	var keys [MaxArgsCount]*C.char
	var vals [MaxArgsCount]*C.char

	if len(a)+len(attr) >= MaxArgsCount {
		return nil, fmt.Errorf("number of keys and vals must be less than %v", MaxArgsCount)
	}

	i := Fillargs(keys[:], vals[:], a)
	if attr != nil {
		for k, v := range attr {
			keys[i] = mxkeys[k]
			vals[i] = Cache(v)
			i++
		}
	}
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
	if e := C.MXSymbolCompose(handle, str, C.uint(len(a)), nil, &a[0]); e != 0 {
		return fmt.Errorf("failed to compose mxnet symbol %v: %v", name, mxLastError())
	}
	return nil
}

const ArgumentsNames = 0
const OutoutNames = 1
const AuxNames = 2

func ListNames(handle SymbolHandle, kind int) ([]string, error) {
	var (
		i      int
		e      C.int
		out_nn C.uint
		out_ns **C.char
		r      []string
	)

	switch kind {
	case ArgumentsNames:
		e = C.MXSymbolListArguments(
			handle,
			&out_nn,
			&out_ns)
	case OutoutNames:
		e = C.MXSymbolListOutputs(
			handle,
			&out_nn,
			&out_ns)
	case AuxNames:
		e = C.MXSymbolListAuxiliaryStates(
			handle,
			&out_nn,
			&out_ns)
	}

	if e != 0 {
		return nil, fmt.Errorf("failed to request output names from mxnet: %v", mxLastError())
	}

	name_at := func(i int) string {
		return C.GoString(*(**C.char)(fu.Index(i, out_ns)))
	}

	r = make([]string, int(out_nn))

	for i = 0; i < int(out_nn); i++ {
		r[i] = name_at(i)
	}

	return r, nil
}

const WithArguments = 1
const WithOutputs = 2
const WithAuxStates = 4
const WithoutOutput = 8

func InferShapes(handle SymbolHandle, with map[string][]int, selector int) (map[string][]int, error) {

	if len(with) > MaxArgsCount {
		return nil, fmt.Errorf("to many shapes in args")
	}

	var (
		keys                  [MaxArgsCount]*C.char
		si                    [MaxArgsCount]C.uint
		sd                    [MaxArgsCount * 4]C.uint
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
		n := int(*(*C.uint)(fu.Index(i, d)))
		r := make([]int, n)
		ps := *(**C.uint)(fu.Index(i, s))

		for j := 0; j < n; j++ {
			r[j] = int(*(*C.int)(fu.Index(j, ps)))
		}

		return r
	}

	r := make(map[string][]int)

	if (selector & WithArguments) != 0 {
		names, err := ListNames(handle, 0)
		if err != nil {
			return nil, err
		}
		for i, name := range names {
			r[name] = shape_at(i, in_sn, in_sd)
		}
	}

	if (selector & WithOutputs) != 0 {
		names, err := ListNames(handle, 1)
		if err != nil {
			return nil, err
		}

		for i, name := range names {
			if _, ok := with[name]; !ok {
				r[name] = shape_at(i, out_sn, out_sd)
			}
		}
	}

	if (selector & WithAuxStates) != 0 {
		names, err := ListNames(handle, 2)
		if err != nil {
			return nil, err
		}

		for i, name := range names {
			if _, ok := with[name]; !ok {
				r[name] = shape_at(i, aux_sn, aux_sd)
			}
		}
	}

	if (selector & WithoutOutput) == 0 {
		r["_output"] = shape_at(0, out_sn, out_sd)
	}

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

func GetInternals(s SymbolHandle) (SymbolHandle, error) {
	var o SymbolHandle
	if e := C.MXSymbolGetInternals(s, &o); e != 0 {
		return nil, fmt.Errorf("failed to get mxnet symbol internals: %v", mxLastError())
	}
	return o, nil
}

func Bind(symbol SymbolHandle, devType, devNo int, args []NDArrayHandle, grads []NDArrayHandle, aux []NDArrayHandle) (ExecutorHandle, error) {
	var r ExecutorHandle

	ga := make([]C.uint, len(args))
	for i := range ga {
		if grads[i] != nil {
			ga[i] = 1
		}
	}

	paux := aux
	if len(aux) == 0 {
		paux = []NDArrayHandle{nil}[:]
	}

	e := C.MXExecutorBind(
		symbol,
		C.int(devType),
		C.int(devNo),
		C.uint(len(args)),
		&args[0],
		&grads[0],
		&ga[0],
		C.uint(len(aux)),
		&paux[0],
		&r)

	if e != 0 {
		return nil, fmt.Errorf("failed to bind mxnet symbols: %v", mxLastError())
	}

	return r, nil
}

type NDArrayInfo struct {
	Handle NDArrayHandle
	Dim    []int
	Type   int
}

func FillInfo(nfo *NDArrayInfo) error {
	var (
		dt C.int
		dn C.uint
		ds *C.uint
	)
	if e := C.MXNDArrayGetDType(nfo.Handle, &dt); e != 0 {
		return fmt.Errorf("failed to get dtype of mxnet ndarray: %v", mxLastError())
	}
	nfo.Type = int(dt)
	if e := C.MXNDArrayGetShape(nfo.Handle, &dn, &ds); e != 0 {
		return fmt.Errorf("failed to get shape of mxnet ndarray: %v", mxLastError())
	}
	nfo.Dim = make([]int, int(dn))
	for i := range nfo.Dim {
		nfo.Dim[i] = int(*(*C.int)(fu.Index(i, ds)))
	}
	return nil
}

func GetOutputs(exec ExecutorHandle) ([]NDArrayInfo, error) {
	var (
		n C.uint
		a *NDArrayHandle
		e C.int
	)
	if e = C.MXExecutorOutputs(exec, &n, &a); e != 0 {
		return nil, fmt.Errorf("failed get mxnet outputs: %v", mxLastError())
	}
	r := make([]NDArrayInfo, int(n))
	for i := range r {
		r[i].Handle = *(*NDArrayHandle)(fu.Index(i, a))
		if err := FillInfo(&r[i]); err != nil {
			return nil, err
		}
	}
	return r, nil
}

func Forward(exec ExecutorHandle, train bool) error {
	t := C.int(0)
	if train {
		t = C.int(1)
	}
	if e := C.MXExecutorForward(exec, t); e != 0 {
		return fmt.Errorf("failed on mxnet forward: %v", mxLastError())
	}
	return nil
}

func Backward(exec ExecutorHandle) error {
	if e := C.MXExecutorBackward(exec, C.uint(0), nil); e != 0 {
		return fmt.Errorf("failed on mxnet forward: %v", mxLastError())
	}
	return nil
}

func OptimizerUpdate(op MxnetOp, params, grads, state1 NDArrayHandle, state2 NDArrayHandle, a ...interface{}) error {
	var keys [MaxArgsCount]*C.char
	var vals [MaxArgsCount]*C.char
	ano := C.int(Fillargs(keys[:], vals[:], a))
	if ent := mxentry[op]; ent != nil {
		if e := C.imperative_invokeN_inout(ent, params, ano, &keys[0], &vals[0], params, grads, state1, state2); e != 0 {
			return fmt.Errorf("mxnet api %v error: %v", op.Value(), mxLastError())
		}
	} else {
		return fmt.Errorf("unresolved API entry %v", op.Value())
	}
	return nil
}

func ToJson(sym SymbolHandle) ([]byte, error) {
	var s *C.char
	if e := C.MXSymbolSaveToJSON(sym, &s); e != 0 {
		return nil, fmt.Errorf("mxnet failed to stringify symbol: %v", mxLastError())
	}
	ln := int(C.strlen(s))
	bs := make([]byte, ln)
	C.memcpy(unsafe.Pointer(&bs[0]), unsafe.Pointer(s), C.ulong(ln))
	return bs, nil
}

func RandomSeed(seed int) error {
	if e := C.MXRandomSeed(C.int(seed)); e != 0 {
		return fmt.Errorf("mxnet failed to set ramdom seed: %v", mxLastError())
	}
	return nil
}

func ContextRandomSeed(seed, devType, devNo int) error {
	if e := C.MXRandomSeedContext(C.int(seed), C.int(devType), C.int(devNo)); e != 0 {
		return fmt.Errorf("mxnet failed to set ramdom seed: %v", mxLastError())
	}
	return nil
}
