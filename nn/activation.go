package nn

import "github.com/sudachen/go-dnn/mx"

func Sigmoid(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.Sigmoid)
}

func Tanh(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.Tanh)
}

func ReLU(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.ReLU)
}

func SoftReLU(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.SoftReLU)
}

func SoftSign(a *mx.Symbol) *mx.Symbol {
	return mx.Activation(a, mx.SoftSign)
}

func Softmax(a *mx.Symbol) *mx.Symbol {
	return mx.SoftmaxActivation(a, false)
}

func ChannelSoftmax(a *mx.Symbol) *mx.Symbol {
	return mx.SoftmaxActivation(a, true)
}
