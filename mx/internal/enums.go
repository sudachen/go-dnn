package internal

type MxnetKey int

const (
	KeyEmpty MxnetKey = iota
	KeyLow
	KeyHigh
	KeyScalar
	KeyLhs
	KeyRhs
	KeyData
	KeyNoKey
)

func (k MxnetKey) Value() string {
	switch k {
	case KeyLow:
		return "low"
	case KeyHigh:
		return "high"
	case KeyScalar:
		return "scalar"
	case KeyLhs:
		return "lhs"
	case KeyRhs:
		return "rhs"
	case KeyData:
		return "data"
	}
	panic("mxnet parameters key out of range")
}

type MxnetOp int

const (
	OpEmpty MxnetOp = iota
	OpRandomUniform
	OpCopyTo
	OpAdd
	OpAddScalar
	OpSub
	OpSubScalar
	OpSubScalarR
	OpMul
	OpMulScalar
	OpDiv
	OpDivScalar
	OpDivScalarR
	OpNoOp
)

var opmap = map[MxnetOp]string{
	OpRandomUniform: "_random_uniform",
	OpCopyTo:        "_copyto",
	OpAdd:           "elemwise_add",
	OpAddScalar:     "_plus_scalar",
	OpSub:           "elemwise_sub",
	OpSubScalar:     "_minus_scalar",
	OpSubScalarR:    "_rminus_scalar",
	OpMul:           "elemwise_mul",
	OpMulScalar:     "_mul_scalar",
	OpDiv:           "elemwise_div",
	OpDivScalar:     "_div_scalar",
	OpDivScalarR:    "_rdiv_scalar",
}

func (o MxnetOp) Value() string {
	if v, ok := opmap[o]; ok {
		return v
	}
	panic("mxnet operation out of range")
}
