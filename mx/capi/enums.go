package capi

type MxnetKey int

const (
	KeyEmpty MxnetKey = iota
	KeyLow
	KeyHigh
	KeyScalar
	KeyLhs
	KeyRhs
	KeyData
	KeyExclude
	KeyAxis
	KeyNormalization
	KeyLr
	KeyMomentum
	KeyWd
	KeyBeta1
	KeyBeta2
	KeyEpsilon
	KeyLoc
	KeyScale
	KeyNoKey
)

var keymap = map[MxnetKey]string{
	KeyLow:           "low",
	KeyHigh:          "high",
	KeyScalar:        "scalar",
	KeyLhs:           "lhs",
	KeyRhs:           "rhs",
	KeyData:          "data",
	KeyExclude:       "exclude",
	KeyAxis:          "axis",
	KeyLr:            "lr",
	KeyMomentum:      "momentum",
	KeyWd:            "wd",
	KeyBeta1:         "beta1",
	KeyBeta2:         "beta2",
	KeyEpsilon:       "epsilon",
	KeyNormalization: "normalization",
	KeyLoc:           "loc",
	KeyScale:         "scale",
}

func (k MxnetKey) Value() string {
	if v, ok := keymap[k]; ok {
		return v
	}
	if k == KeyEmpty {
		return ""
	}
	panic("mxnet parameters key out of range")
}

type MxnetOp int

const (
	OpEmpty MxnetOp = iota
	OpRandomUniform
	OpRandomNormal
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
	OpPowerScalar
	OpPowerScalarR
	OpSgdUpdate
	OpAdamUpdate
	OpNoOp
)

var opmap = map[MxnetOp]string{
	OpRandomUniform: "_random_uniform",
	OpRandomNormal:  "_random_normal",
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
	OpPowerScalar:   "_power_scalar",
	OpPowerScalarR:  "_rpower_scalar",
	OpSgdUpdate:     "sgd_mom_update",
	OpAdamUpdate:    "adam_update",
}

func (o MxnetOp) Value() string {
	if v, ok := opmap[o]; ok {
		return v
	}
	panic("mxnet operation out of range")
}
