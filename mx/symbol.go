package mx

import (
	"fmt"
)

type SymbolOp int

const (
	NoneOp SymbolOp = iota
	VarOp
	InputOp
	JsonOp
	ScalarOp
	AddOp
	SubOp
	MulOp
	DivOp
	DotOp
)

type Symbol struct {
	op    SymbolOp
	value string
	l, r  *Symbol
}

func (s *Symbol) String() string {
	return "op"
	//return fmt.Sprintf("&op{%s}",s.name)
}

type _input_ struct{}

func Input(_input_) {}

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

func Dot(lv interface{}, rv interface{}) *Symbol {
	return GenericOp2(DotOp, lv, rv)
}

func Var(name string) *Symbol {
	return &Symbol{op: VarOp, value: name}
}

func SymbolCast(i interface{}) (*Symbol, error) {
	var o *Symbol
	switch v := i.(type) {
	case func(_input_):
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
	return &Symbol{op: op, l: l, r: r}
}
