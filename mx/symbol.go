package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx/capi"
	"strings"
)

const (
	OpVar_    capi.MxnetOp = -1
	OpInput_  capi.MxnetOp = -2
	OpScalar_ capi.MxnetOp = -4
	OpNogVar_ capi.MxnetOp = -5
	OpGroup_  capi.MxnetOp = -7
	OpRef_    capi.MxnetOp = -8
	OpOutput_ capi.MxnetOp = -9
	OpBound_  capi.MxnetOp = -10
	OpDepend_ capi.MxnetOp = -11
)

type Inite interface {
	Inite(*NDArray) error
}

type _Value struct { Value []float32 }
func (v *_Value) Inite(arr *NDArray) error {
	return arr.SetValues(v.Value)
}

var _symbolId = 0

func NextSymbolId() int {
	_symbolId++
	return _symbolId
}

func ResetSymbolId(first int) {
	_symbolId = first
}

type Symbol struct {
	op    capi.MxnetOp
	value string
	name  string
	args  []*Symbol
	init  Inite
	attr  map[capi.MxnetKey]string
	dim   Dimension
	output bool
}

type _hidden_input_ struct{}

func Input(..._hidden_input_) *Symbol { return &Symbol{op: OpInput_} }

type _hidden_nograd_ struct{}

func Nograd(_hidden_nograd_) {}

func (s *Symbol) SetName(name string) *Symbol {
	s.name = name
	return s
}

func (s *Symbol) SetOutput(on bool) *Symbol {
	s.output = on
	return s
}

func Output(a *Symbol, name string) *Symbol {
	return &Symbol{op:OpOutput_, args:[]*Symbol{a}, name: name}
}

func Bound(a ...*Symbol) *Symbol {
	return &Symbol{op:OpBound_, args:a}
}

func Depend(a ...*Symbol) *Symbol {
	return &Symbol{op:OpDepend_, args:a}
}

func SymbolCast(i interface{}) (*Symbol, error) {
	var o *Symbol
	switch v := i.(type) {
	case func(..._hidden_input_):
		o = &Symbol{op: OpInput_}
	case string:
		o = Var(v)
	case *Symbol:
		o = v
	case float32, float64, int, int8, int32, int64, uint, uint8, uint32, uint64:
		o = &Symbol{op: OpScalar_, value: fmt.Sprintf("%v", v)}
	}
	if o != nil {
		return o, nil
	}
	return nil, fmt.Errorf("cant cast '%#v' to *Symbol", i)
}

func GenericOp2(op, opScalar, opScalarR capi.MxnetOp, lv interface{}, rv interface{}) *Symbol {
	var (
		l, r *Symbol
		err  error
	)
	if l, err = SymbolCast(lv); err != nil {
		panic(err.Error())
	}
	if r, err = SymbolCast(rv); err != nil {
		panic(err.Error())
	}

	if l != nil && l.op == OpScalar_ {
		return &Symbol{
			op:   opScalarR,
			args: []*Symbol{r},
			attr: map[capi.MxnetKey]string{capi.KeyScalar: l.value}}
	}

	if r != nil && r.op == OpScalar_ {
		return &Symbol{
			op:   opScalar,
			args: []*Symbol{l},
			attr: map[capi.MxnetKey]string{capi.KeyScalar: r.value}}
	}

	return &Symbol{op: op, args: []*Symbol{l, r}}
}

func GenericOp1(op, opScalar capi.MxnetOp, l *Symbol, rv interface{}) *Symbol {
	var (
		r   *Symbol
		err error
	)
	if r, err = SymbolCast(rv); err != nil {
		panic(err.Error())
	}

	if r != nil && r.op == OpScalar_ {
		return &Symbol{
			op:   opScalar,
			args: []*Symbol{l},
			attr: map[capi.MxnetKey]string{capi.KeyScalar: r.value}}
	}

	return &Symbol{op: op, args: []*Symbol{l, r}}
}

func Add(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpAdd, capi.OpAddScalar, capi.OpAddScalar, lv, rv)
}

func Sub(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpSub, capi.OpSubScalar, capi.OpSubScalarR, lv, rv)
}

func Mul(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpMul, capi.OpMulScalar, capi.OpMulScalar, lv, rv)
}

func Div(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpDiv, capi.OpDivScalar, capi.OpDivScalarR, lv, rv)
}

func Dot(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpDot, capi.OpEmpty, capi.OpEmpty, lv, rv)
}

func LE(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpLe, capi.OpLeScalar, a, rv)
}

func GE(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpGe, capi.OpGeScalar, a, rv)
}

func EQ(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpEq, capi.OpEqScalar, a, rv)
}

func NE(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpNe, capi.OpNeScalar, a, rv)
}

func Lesser(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpLesser, capi.OpLesserScalar, a, rv)
}

func Greater(a *Symbol, rv interface{}) *Symbol {
	return GenericOp1(capi.OpGreater, capi.OpGreaterScalar, a, rv)
}

func And(a *Symbol, b *Symbol) *Symbol {
	return &Symbol{op:capi.OpAnd, args: []*Symbol{a,b}}
}

func Or(a *Symbol, b *Symbol) *Symbol {
	return &Symbol{op:capi.OpOr, args: []*Symbol{a,b}}
}

func Xor(a *Symbol, b *Symbol) *Symbol {
	return &Symbol{op:capi.OpXor, args: []*Symbol{a,b}}
}

func BcastAdd(a, b *Symbol) *Symbol {
	return &Symbol{
		op:   capi.OpBroadcastAdd,
		args: []*Symbol{a, b},
	}
}

func BcastMul(a, b *Symbol) *Symbol {
	return &Symbol{
		op:   capi.OpBroadcastMul,
		args: []*Symbol{a, b},
	}
}

func BcastDiv(a, b *Symbol) *Symbol {
	return &Symbol{
		op:   capi.OpBroadcastDiv,
		args: []*Symbol{a, b},
	}
}

func BcastSub(a, b *Symbol) *Symbol {
	return &Symbol{
		op:   capi.OpBroadcastSub,
		args: []*Symbol{a, b},
	}
}

func Log(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpLog, args: []*Symbol{a}}
}

func Cosh(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpCosh, args: []*Symbol{a}}
}

func LogCosh(a *Symbol) *Symbol {
	return Log(Cosh(a))
}

func Not(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpNot, args: []*Symbol{a}}
}

func Var(name string, opt ...interface{}) *Symbol {
	s := &Symbol{op: OpVar_, name: name}
	for _, t := range opt {
		if t == nil {
			continue
		} else if init, ok := t.(Inite); ok {
			s.init = init
		} else if _, ok := t.(func(_hidden_nograd_)); ok {
			s.op = OpNogVar_
		} else if dim, ok := t.(Dimension); ok {
			s.dim = dim
		} else {
			panic(fmt.Sprintf("unexpected parameter %v", t))
		}
	}
	return s
}

func Value(name string, a ...float32) *Symbol{
	return Var(name, Dim(len(a)), &_Value{Value: a})
}

func Ref(name string, a ...*Symbol) *Symbol {
	return &Symbol{op: OpRef_, name: name, args: a}
}

func Group(a ...*Symbol) *Symbol {
	return &Symbol{op: OpGroup_, args: a}
}

func MakeLoss(s *Symbol) *Symbol {
	return &Symbol{op: capi.OpMakeLoss, args: []*Symbol{s}}
}

func BlockGrad(s *Symbol) *Symbol {
	return &Symbol{op: capi.OpBlockGrad, args: []*Symbol{s}}
}

func Pow(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(capi.OpEmpty, capi.OpPowerScalar, capi.OpPowerScalarR, lv, rv)
}

func Abs(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpAbs, args: []*Symbol{a}}
}

func Square(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpSquare, args: []*Symbol{a}}
}

func Sqrt(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpSqrt, args: []*Symbol{a}}
}

func Minus(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpMulScalar, args: []*Symbol{a},
		attr: map[capi.MxnetKey]string{capi.KeyScalar: "-1"}}
}

func Pick(a *Symbol, label *Symbol) *Symbol {
	return &Symbol{op: capi.OpPick, args: []*Symbol{a, label},
		attr: map[capi.MxnetKey]string{capi.KeyKeepdims: "1"}}
}

func LogSoftmax(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpLogSoftmax, args: []*Symbol{a}}
	if len(axis) >= 1 {
		s.attr = map[capi.MxnetKey]string{capi.KeyAxis: fmt.Sprintf("%v", axis[0])}
	}
	return s
}

func SoftmaxOutput(a *Symbol, l *Symbol, multiOutput bool) *Symbol {
	s := &Symbol{op: capi.OpSoftmaxOutput, args: []*Symbol{a, l}, attr: map[capi.MxnetKey]string{}}
	if multiOutput {
		s.attr[capi.KeyMultiOutput] = "1"
	}
	return s
}

func Softmax(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpSoftmax, args: []*Symbol{a}}
	if len(axis) >= 1 {
		s.attr = map[capi.MxnetKey]string{capi.KeyAxis: fmt.Sprintf("%v", axis[0])}
	}
	return s
}

func SoftmaxActivation(a *Symbol, channel bool) *Symbol {
	s := &Symbol{op: capi.OpSoftmaxAC, args: []*Symbol{a}}
	if channel {
		s.attr = map[capi.MxnetKey]string{capi.KeyMode: "channel"}
	}
	return s
}

func SoftmaxCrossEntropy(a, b *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpSoftmaxCE, args: []*Symbol{a, b}}
	if len(axis) >= 1 {
		s.attr = map[capi.MxnetKey]string{capi.KeyAxis: fmt.Sprintf("%v", axis[0])}
	}
	return s
}

func formatAxis(axis ...int) string {
	if len(axis) == 1 {
		switch axis[0] {
		case 0:
			return "0"
		case 1:
			return "1"
		case -1:
			return "-1"
		default:
			return fmt.Sprintf("%d", axis[0])
		}
	} else {
		s := make([]string, len(axis))
		for i, a := range axis {
			switch a {
			case 0:
				s[i] = "0"
			case 1:
				s[i] = "1"
			case -1:
				s[i] = "-1"
			default:
				s[i] = fmt.Sprintf("%d", a)
			}
		}
		return "(" + strings.Join(s, ",") + ")"
	}
}

func SumNan(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpSumNan, args: []*Symbol{a}}
	if len(axis) > 0 {
		s.attr = map[capi.MxnetKey]string{
			capi.KeyAxis: formatAxis(axis...),
		}
	}
	return s
}

func Sum(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpSum, args: []*Symbol{a}}
	if len(axis) > 0 {
		s.attr = map[capi.MxnetKey]string{
			capi.KeyAxis: formatAxis(axis...),
		}
	}
	return s
}

func Sum1(a *Symbol) *Symbol {
	s := &Symbol{op: capi.OpSum, args: []*Symbol{a}}
	s.attr = map[capi.MxnetKey]string{
		capi.KeyAxis: "-1",
		capi.KeyKeepdims: "1",
	}
	return s
}

func SumXl(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpSum, args: []*Symbol{a}}
	if len(axis) > 0 {
		s.attr = map[capi.MxnetKey]string{
			capi.KeyExclude: "1",
			capi.KeyAxis:    formatAxis(axis...),
		}
	}
	return s
}

func Mean(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpMean, args: []*Symbol{a}}
	if len(axis) > 0 {
		s.attr = map[capi.MxnetKey]string{
			capi.KeyAxis: formatAxis(axis...),
		}
	}
	return s
}

func MeanKd(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpMean, args: []*Symbol{a},
		attr: map[capi.MxnetKey]string{
			capi.KeyKeepdims: "1",
		}}
	if len(axis) > 0 {
		s.attr[capi.KeyAxis] = formatAxis(axis...)
	}
	return s
}

func MeanXl(a *Symbol, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpMean, args: []*Symbol{a}}
	if len(axis) > 0 {
		s.attr = map[capi.MxnetKey]string{
			capi.KeyExclude: "1",
			capi.KeyAxis:    formatAxis(axis...),
		}
	}
	return s
}

func Stack(a ...*Symbol) *Symbol {
	s := &Symbol{op: capi.OpStack, args: a,
		attr: map[capi.MxnetKey]string{
			capi.KeyNumArgs: fmt.Sprintf("%d", len(a)),
		}}
	return s
}

func Stack1(a ...*Symbol) *Symbol {
	s := &Symbol{op: capi.OpStack, args: a,
		attr: map[capi.MxnetKey]string{
			capi.KeyNumArgs: fmt.Sprintf("%d", len(a)),
			capi.KeyAxis:    "-1",
		}}
	return s
}

func BatchNorm(a, gamma, beta, rmean, rvar *Symbol, mom, eps float32, useGlobalStats bool, axis ...int) *Symbol {
	s := &Symbol{op: capi.OpBatchNorm, args: []*Symbol{a, gamma, beta, rmean, rvar}}
	s.attr = map[capi.MxnetKey]string{}
	if len(axis) > 0 {
		s.attr[capi.KeyAxis] = formatAxis(axis...)
	}
	if mom != 0 {
		s.attr[capi.KeyMomentum] = fmt.Sprintf("%v", mom)
	}
	if eps != 0 {
		s.attr[capi.KeyEps] = fmt.Sprintf("%v", eps)
	}
	if useGlobalStats {
		s.attr[capi.KeyGlobalStats] = "1"
	}
	return s
}

func Concat(a ...*Symbol) *Symbol {
	return &Symbol{op: capi.OpConcat, args: a,
		attr: map[capi.MxnetKey]string{capi.KeyNumArgs: fmt.Sprintf("%d", len(a))}}
}

func Conv(a, weight, bias *Symbol, channels int, kernel, stride, padding Dimension, groups bool, layout string) *Symbol {
	args := []*Symbol{a, weight, bias}
	attr := map[capi.MxnetKey]string{capi.KeyNumFilter: fmt.Sprintf("%v", channels)}
	if bias == nil {
		attr[capi.KeyNoBias] = "1"
	}
	if groups {
		attr[capi.KeyNumGroup] = "2"
	}

	if kernel.Len > 1 {
		attr[capi.KeyKernel] = kernel.String()
	} else if kernel.Len == 1 {
		attr[capi.KeyKernel] = fmt.Sprintf("%v", kernel.Shape[0])
	}

	if stride.Len > 1 {
		attr[capi.KeyStride] = stride.String()
	} else if stride.Len == 1 {
		attr[capi.KeyStride] = fmt.Sprintf("%v", stride.Shape[0])
	}

	if padding.Len > 1 {
		attr[capi.KeyPad] = padding.String()
	} else if padding.Len == 1 {
		attr[capi.KeyPad] = fmt.Sprintf("%v", padding.Shape[0])
	}

	if layout != "" {
		attr[capi.KeyLayout] = layout
	}

	return &Symbol{op: capi.OpConvolution, args: args, attr: attr}
}

type ActivationType int

const (
	ActivReLU ActivationType = iota
	ActivSoftReLU
	ActivSoftSign
	ActivSigmoid
	ActivTanh
)

func Activation(a *Symbol, actType ActivationType) *Symbol {
	var s string
	switch actType {
	case ActivSoftReLU:
		s = "softrelu"
	case ActivSoftSign:
		s = "softsign"
	case ActivSigmoid:
		s = "sigmoid"
	case ActivTanh:
		s = "tanh"
	//case ReLU: s = "relu"
	default:
		s = "relu"
	}
	return &Symbol{op: capi.OpActivation, args: []*Symbol{a},
		attr: map[capi.MxnetKey]string{capi.KeyActType: s}}
}

func Pool(a *Symbol, kernel, stride, padding Dimension, ceil bool, maxpool bool) *Symbol {
	attr := map[capi.MxnetKey]string{}

	if kernel.Len > 1 {
		attr[capi.KeyKernel] = kernel.String()
	} else if kernel.Len == 1 {
		attr[capi.KeyKernel] = fmt.Sprintf("%v", kernel.Shape[0])
	}

	if stride.Len > 1 {
		attr[capi.KeyStride] = stride.String()
	} else if stride.Len == 1 {
		attr[capi.KeyStride] = fmt.Sprintf("%v", stride.Shape[0])
	}

	if padding.Len > 1 {
		attr[capi.KeyPad] = padding.String()
	} else if padding.Len == 1 {
		attr[capi.KeyPad] = fmt.Sprintf("%v", padding.Shape[0])
	}

	if maxpool {
		attr[capi.KeyPoolType] = "max"
	} else {
		attr[capi.KeyPoolType] = "avg"
	}

	if ceil {
		attr[capi.KeyPoolConv] = "full"
	} else {
		attr[capi.KeyPoolConv] = "valid"
	}

	return &Symbol{op: capi.OpPooling, args: []*Symbol{a}, attr: attr}
}

func FullyConnected(a, weight, bias *Symbol, size int, flatten bool) *Symbol {
	args := []*Symbol{a, weight, bias}
	attr := map[capi.MxnetKey]string{}
	if bias == nil {
		attr[capi.KeyNoBias] = "1"
	}
	if flatten {
		attr[capi.KeyFlatten] = "1"
	}
	attr[capi.KeyNumHidden] = fmt.Sprintf("%v", size)
	return &Symbol{op: capi.OpFullyConnected, args: args, attr: attr}
}

func Flatten(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpFlatten, args: []*Symbol{a}}
}

func Sigmoid(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpSigmoid, args: []*Symbol{a}}
}

func HardSigmoid(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpHardSigmoid, args: []*Symbol{a}}
}

func Tanh(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpTanh, args: []*Symbol{a}}
}

func Sin(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpSin, args: []*Symbol{a}}
}

func ReLU(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpReLU, args: []*Symbol{a}}
}

func Transpose(a *Symbol, axis ...int) *Symbol {
	s := make([]string, len(axis))
	for i, a := range axis {
		switch a {
		case 0:
			s[i] = "0"
		case 1:
			s[i] = "1"
		case -1:
			s[i] = "-1"
		default:
			s[i] = fmt.Sprintf("%d", a)
		}
	}
	ax := "(" + strings.Join(s, ",") + ")"
	return &Symbol{
		op:   capi.OpTranspose,
		args: []*Symbol{a},
		attr: map[capi.MxnetKey]string{capi.KeyAxes: ax}}
}

func Slice(a *Symbol, axis, begin, end int) *Symbol {
	s := "("
	for i := 0; i < axis; i++ {
		s += "None,"
	}
	return &Symbol{
		op:   capi.OpSlice,
		args: []*Symbol{a},
		attr: map[capi.MxnetKey]string{
			capi.KeyBegin: fmt.Sprintf(s+"%d)", begin),
			capi.KeyEnd:   fmt.Sprintf(s+"%d)", end),
		}}
}

func Channel(a *Symbol, ch int) *Symbol {
	return &Symbol{
		op:   capi.OpSlice,
		args: []*Symbol{a},
		attr: map[capi.MxnetKey]string{
			capi.KeyBegin: fmt.Sprintf("(None,%d)", ch),
			capi.KeyEnd:   fmt.Sprintf("(None,%d)", ch+1),
		}}
}

func Ones(dim ...int) *Symbol {
	return &Symbol{
		op:  capi.OpOnes,
		dim: Dim(dim...),
	}
}

func OnesLike(a *Symbol) *Symbol {
	return &Symbol{
		op:   capi.OpOnesLike,
		args: []*Symbol{a},
	}
}

func Zeros(dim ...int) *Symbol {
	return &Symbol{
		op:  capi.OpZeros,
		dim: Dim(dim...),
	}
}

func ZerosLike(a *Symbol) *Symbol {
	return &Symbol{
		op:   capi.OpZerosLike,
		args: []*Symbol{a},
	}
}

func ReshapeLike(a, b *Symbol) *Symbol {
	return &Symbol{
		op:   capi.OpReshapeLike,
		args: []*Symbol{a, b},
	}
}

func Dropout(a *Symbol, rate float32) *Symbol {
	return &Symbol{
		op:   capi.OpDropout,
		args: []*Symbol{a},
		attr: map[capi.MxnetKey]string{
			capi.KeyP: fmt.Sprintf("%v", rate),
		},
	}
}