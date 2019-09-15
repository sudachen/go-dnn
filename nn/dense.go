package nn

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
)

type Flatten struct{}

func (ly *Flatten) Combine(a *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	return mx.Flatten(a), g, nil
}

type FullyConnected struct {
	Size       int
	Activation func(*mx.Symbol) *mx.Symbol
	WeightInit mx.Inite // none by default
	BiasInit   mx.Inite // &nn.Const{0} by default
	NoBias     bool
	NoFlatten  bool
	Name       string
}

func (ly *FullyConnected) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	var (
		out, weight, bias *mx.Symbol
	)
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("FullyConnected%02d", mx.NextSymbolId())
	}
	weight = mx.Var(ns+"_weight", ly.WeightInit)
	if !ly.NoBias {
		init := ly.BiasInit
		if init == nil {
			init = &Const{0}
		}
		bias = mx.Var(ns+"_bias", init)
	}
	out = mx.FullyConnected(in, weight, bias, ly.Size, !ly.NoFlatten)
	out.SetName(ns)
	if ly.Activation != nil {
		out = ly.Activation(out)
	}
	out = out
	return out, g, nil
}
