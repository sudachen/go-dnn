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
	KeyMode
	KeyNormalization
	KeyLr
	KeyMomentum
	KeyWd
	KeyBeta1
	KeyBeta2
	KeyEpsilon
	KeyEps
	KeyLoc
	KeyScale
	KeyKeepdims
	KeyNoBias
	KeyNumGroup
	KeyNumFilter
	KeyKernel
	KeyStride
	KeyPad
	KeyActType
	KeyPoolType
	KeyPoolConv
	KeyFlatten
	KeyNumHidden
	KeyMultiOutput
	KeyNumArgs
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
	KeyMode:          "mode",
	KeyLr:            "lr",
	KeyMomentum:      "momentum",
	KeyWd:            "wd",
	KeyBeta1:         "beta1",
	KeyBeta2:         "beta2",
	KeyEpsilon:       "epsilon",
	KeyEps:           "eps",
	KeyNormalization: "normalization",
	KeyLoc:           "loc",
	KeyScale:         "scale",
	KeyKeepdims:      "keepdims",
	KeyNoBias:        "no_bias",
	KeyNumGroup:      "num_group",
	KeyNumFilter:     "num_filter",
	KeyKernel:        "kernel",
	KeyStride:        "stride",
	KeyPad:           "pad",
	KeyActType:       "act_type",
	KeyPoolType:      "pool_type",
	KeyPoolConv:      "pooling_convention",
	KeyFlatten:       "flatten",
	KeyMultiOutput:   "multi_output",
	KeyNumHidden:     "num_hidden",
	KeyNumArgs:       "num_args",
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
	OpStack
	OpAbs
	OpBlockGrad
	OpMakeLoss
	OpZeros
	OpPowerScalar
	OpPowerScalarR
	OpSgdUpdate
	OpSgdMomUpdate
	OpAdamUpdate
	OpLogSoftmax
	OpSoftmax
	OpSoftmaxOutput
	OpSoftmaxCE
	OpSoftmaxAC
	OpSum
	OpDot
	OpPick
	OpSquare
	OpConcat
	OpConvolution
	OpActivation
	OpPooling
	OpFullyConnected
	OpFlatten
	OpLog
	OpCosh
	OpNot
	OpSigmoid
	OpTanh
	OpSin
	OpBatchNorm
	OpNoOp
)

var opmap = map[MxnetOp]string{
	OpRandomUniform:  "_random_uniform",
	OpRandomNormal:   "_random_normal",
	OpCopyTo:         "_copyto",
	OpAdd:            "elemwise_add",
	OpAddScalar:      "_plus_scalar",
	OpSub:            "elemwise_sub",
	OpSubScalar:      "_minus_scalar",
	OpSubScalarR:     "_rminus_scalar",
	OpMul:            "elemwise_mul",
	OpMulScalar:      "_mul_scalar",
	OpDiv:            "elemwise_div",
	OpDivScalar:      "_div_scalar",
	OpDivScalarR:     "_rdiv_scalar",
	OpMean:           "mean",
	OpStack:          "stack",
	OpAbs:            "abs",
	OpBlockGrad:      "BlockGrad",
	OpMakeLoss:       "make_loss",
	OpZeros:          "_zeros",
	OpPowerScalar:    "_power_scalar",
	OpPowerScalarR:   "_rpower_scalar",
	OpSgdUpdate:      "sgd_update",
	OpSgdMomUpdate:   "sgd_mom_update",
	OpAdamUpdate:     "adam_update",
	OpLogSoftmax:     "log_softmax",
	OpSoftmax:        "softmax",
	OpSoftmaxCE:      "softmax_cross_entropy",
	OpSoftmaxAC:      "SoftmaxActivation",
	OpSoftmaxOutput:  "SoftmaxOutput",
	OpSum:            "sum",
	OpDot:            "dot",
	OpPick:           "pick",
	OpSquare:         "square",
	OpConcat:         "Concat",
	OpConvolution:    "Convolution",
	OpActivation:     "Activation",
	OpPooling:        "Pooling",
	OpFullyConnected: "FullyConnected",
	OpFlatten:        "Flatten",
	OpNot:            "logical_not",
	OpLog:            "log",
	OpCosh:           "cosh",
	OpSin:            "sin",
	OpTanh:           "tanh",
	OpSigmoid:        "sigmoid",
	OpBatchNorm:      "BatchNorm",
}

func (o MxnetOp) Value() string {
	if v, ok := opmap[o]; ok {
		return v
	}
	panic("mxnet operation out of range")
}
