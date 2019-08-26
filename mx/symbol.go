package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx/capi"
)

const (
	OpVar_    capi.MxnetOp = -1
	OpInput_  capi.MxnetOp = -2
	OpScalar_ capi.MxnetOp = -4
	OpNogVar_ capi.MxnetOp = -5
	OpGroup_  capi.MxnetOp = -7
)

type Inite interface {
	Inite(*NDArray) error
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
}

type _hidden_input_ struct{}

func Input(..._hidden_input_) *Symbol { return &Symbol{op: OpInput_} }

type _hidden_nograd_ struct{}

func Nograd(_hidden_nograd_) {}

func (s *Symbol) SetName(name string) *Symbol {
	s.name = name
	return s
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

func Log(a *Symbol) *Symbol {
	return &Symbol{op: capi.OpLog, args: []*Symbol{a}}
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
		} else {
			panic(fmt.Sprintf("unexpected parameter %v", t))
		}
	}
	return s
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

func Sum(a *Symbol, batch bool) *Symbol {
	s := &Symbol{op: capi.OpSum, args: []*Symbol{a}}
	if batch {
		s.attr = map[capi.MxnetKey]string{
			capi.KeyExclude: "1",
			capi.KeyAxis:    "0",
		}
	}
	return s
}

func Mean(a *Symbol, batch bool) *Symbol {
	s := &Symbol{op: capi.OpMean, args: []*Symbol{a}}
	if batch {
		s.attr = map[capi.MxnetKey]string{
			capi.KeyExclude: "1",
			capi.KeyAxis:    "0",
		}
	}
	return s
}

func Concat(a ...*Symbol) *Symbol {
	return &Symbol{op: capi.OpConcat, args: a}
}

func Conv(a, weight, bias *Symbol, channels int, kernel, stride, padding Dimension, groups bool) *Symbol {
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

	return &Symbol{op: capi.OpConvolution, args: args, attr: attr}
}

type ActivationType int

const (
	ReLU ActivationType = iota
	SoftReLU
	SoftSign
	Sigmoid
	Tanh
)

func Activation(a *Symbol, actType ActivationType) *Symbol {
	var s string
	switch actType {
	case SoftReLU:
		s = "softrelu"
	case SoftSign:
		s = "softsign"
	case Sigmoid:
		s = "sigmoid"
	case Tanh:
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
