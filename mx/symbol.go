package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx/capi"
)

type SymbolOp int

const BinaryOpFlag SymbolOp = 0x100000

const (
	VarOp    SymbolOp = -1
	InputOp  SymbolOp = -2
	JsonOp   SymbolOp = -3
	ScalarOp SymbolOp = -4
	AgVarOp  SymbolOp = -5
	NsOp     SymbolOp = -6
)

const (
	NoneOp SymbolOp = iota
	AddOp
	SubOp
	MulOp
	DivOp
	DotOp
	MeanOp
	AbsOp
	GroupOp
	MakeLossOp
	BlockGradOp
	PowOp
)

type VarInitializer = func(*NDArray) error

type Symbol struct {
	op    SymbolOp
	value string
	args  []*Symbol
	init  VarInitializer
	attr  map[capi.MxnetKey]string
}

func (s *Symbol) String() string {
	return "op"
	//return fmt.Sprintf("&op{%s}",s.name)
}

type _hidden_input_ struct{}

func Input(..._hidden_input_) *Symbol { return &Symbol{op: InputOp} }

type _hidden_autograd_ struct{}

func Autograd(_hidden_autograd_) {}

func JsonSymbol(json string) *Symbol {
	return &Symbol{op: JsonOp, value: json}
}

func Ns(ns string, s *Symbol) *Symbol {
	return &Symbol{op: NsOp, value: ns, args: []*Symbol{s}}
}

func Add(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(AddOp, lv, rv)
}

func Sub(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(SubOp, lv, rv)
}

func Mul(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(MulOp, lv, rv)
}

func Div(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(DivOp, lv, rv)
}

func Dot(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(DotOp, lv, rv)
}

func Var(name string, opt ...interface{}) *Symbol {
	s := &Symbol{op: VarOp, value: name}
	for _, t := range opt {
		if init, ok := t.(func(*NDArray) error); ok {
			s.init = init
		}
		if _, ok := t.(func(_hidden_autograd_)); ok {
			s.op = AgVarOp
		}
	}
	return s
}

func Group(a ...*Symbol) *Symbol {
	return &Symbol{op: GroupOp, args: a}
}

func MakeLoss(s *Symbol) *Symbol {
	return &Symbol{op: MakeLossOp, args: []*Symbol{s}}
}

func BlockGrad(s *Symbol) *Symbol {
	return &Symbol{op: BlockGradOp, args: []*Symbol{s}}
}

func Pow(s *Symbol, rv interface{}) *Symbol {
	return GenericOp2(PowOp, s, rv)
}

//func ReshapeLike(a *Symbol, b *Symbol) *Symbol {
//	return nil
//}

func Abs(a *Symbol) *Symbol {
	return &Symbol{
		op:   AbsOp,
		args: []*Symbol{a},
	}
}

func Mean(a *Symbol) *Symbol {
	return &Symbol{
		op:   MeanOp,
		args: []*Symbol{a},
		attr: map[capi.MxnetKey]string{
			capi.KeyExclude: "1",
			capi.KeyAxis:    "0",
		},
	}
}

func SymbolCast(i interface{}) (*Symbol, error) {
	var o *Symbol
	switch v := i.(type) {
	case func(..._hidden_input_):
		o = &Symbol{op: InputOp}
	case string:
		o = Var(v)
	case *Symbol:
		o = v
	case float32, float64, int, int8, int32, int64, uint, uint8, uint32, uint64:
		o = &Symbol{op: ScalarOp, value: fmt.Sprintf("%v", v)}
	}
	if o != nil {
		return o, nil
	}
	return nil, fmt.Errorf("cant cast '%#v' to *Symbol", i)
}

func GenericOp2(op SymbolOp, lv interface{}, rv interface{}) *Symbol {
	var l, r *Symbol
	var err error
	if l, err = SymbolCast(lv); err != nil {
		panic(err.Error())
	}
	if r, err = SymbolCast(rv); err != nil {
		panic(err.Error())
	}
	args := [2]*Symbol{l, r}
	return &Symbol{op: op, args: args[:]}
}
