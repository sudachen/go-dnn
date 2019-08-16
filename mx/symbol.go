package mx

import (
	"fmt"
)

type SymbolOp int

const BinaryOpFlag SymbolOp = 0x100000

const (
	VarOp    SymbolOp = -1
	InputOp  SymbolOp = -2
	JsonOp   SymbolOp = -3
	ScalarOp SymbolOp = -4
)

const (
	NoneOp SymbolOp = iota
	AddOp
	SubOp
	MulOp
	DivOp
	DotOp
)

type Symbol struct {
	op    SymbolOp
	value string
	args  []*Symbol
}

func (s *Symbol) String() string {
	return "op"
	//return fmt.Sprintf("&op{%s}",s.name)
}

type _input_ struct{}

func Input(..._input_) *Symbol { return &Symbol{op: InputOp} }

func JsonSymbol(json string) *Symbol {
	return &Symbol{op: JsonOp, value: json}
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

func Var(name string) *Symbol {
	return &Symbol{op: VarOp, value: name}
}

func SymbolCast(i interface{}) (*Symbol, error) {
	var o *Symbol
	switch v := i.(type) {
	case func(..._input_):
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
	return nil, fmt.Errorf("cant cast '%#v' to *Operation", i)
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
