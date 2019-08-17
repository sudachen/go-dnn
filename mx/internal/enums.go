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
	KeyExclude
	KeyAxis
	KeyNormalization
)

var keymap = map[MxnetKey]string{
	KeyLow:           "low",
	KeyHigh:          "high",
	KeyScalar:        "scalar",
	KeyLhs:           "lhs",
	KeyRhs:           "rhs",
	KeyData:          "data",
	KeyExclude:       "excludd",
	KeyAxis:          "axis",
	KeyNormalization: "normalization",
}

func (k MxnetKey) Value() string {
	if v, ok := keymap[k]; ok {
		return v
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
	OpMean
	OpAbs
	OpGroup
	OpBlockGrad
	OpMakeLoss
	OpZeros
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
	OpMean:          "mean",
	OpAbs:           "abs",
	OpGroup:         "group",
	OpBlockGrad:     "BlockGrad",
	OpMakeLoss:      "make_loss",
	OpZeros:         "_zeros",
}

func (o MxnetOp) Value() string {
	if v, ok := opmap[o]; ok {
		return v
	}
	panic("mxnet operation out of range")
}
