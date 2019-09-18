package nn

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
)

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

func Swish(a *mx.Symbol) *mx.Symbol {
	return mx.Mul(mx.Sigma(a),a)
}

type Activation struct {
	Function func(*mx.Symbol) *mx.Symbol
	BatchNorm bool
	Name string
}

func (ly *Activation) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	var err error
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("Activation%02d", mx.NextSymbolId())
	} else {
		ns += "$A"
	}
	out := in
	if ly.BatchNorm {
		if out, g, err = (&BatchNorm{Name: ly.Name}).Combine(in, g...); err != nil {
			return nil, nil, err
		}
	}
	if ly.Function != nil {
		out = ly.Function(out)
	}
	out.SetName(ns)
	return out, g, nil
}

type BatchNorm struct {
	Name string
	Mom, Epsilon float32
}

func (ly *BatchNorm) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("BatchNorm%02d", mx.NextSymbolId())
	} else {
		ns += "$BN"
	}

	gamma := mx.Var(ns+"_gamma", &Const{1})
	beta := mx.Var(ns+"_beta", &Const{0})
	running_mean := mx.Var(ns+"_rmean", mx.Nograd, &Const{0})
	running_var := mx.Var(ns+"_rvar", mx.Nograd, &Const{1})
	out := mx.BatchNorm(in, gamma, beta, running_mean, running_var, ly.Mom, ly.Epsilon)
	out.SetName(ns)
	return out, g, nil
}
