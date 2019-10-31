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
	BatchNorm  bool
	Name       string
	Output     bool
	Dropout    float32
}

func (ly *FullyConnected) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	var (
		out, weight, bias *mx.Symbol
		err               error
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
	if ly.BatchNorm {
		if out, g, err = (&BatchNorm{Name: ns}).Combine(out, g...); err != nil {
			return nil, nil, err
		}
	}
	if ly.Activation != nil {
		out = ly.Activation(out)
		out.SetName(ns + "$A")
	}
	if ly.Dropout > 0.01 {
		out = mx.Dropout(out,ly.Dropout)
		out.SetName(ns + "$D")
	}
	out.SetOutput(ly.Output)
	return out, g, nil
}

type Dropout struct {
	Rate float32
}

func (ly *Dropout) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	out := in
	if ly.Rate > 0.01 {
		out = mx.Dropout(out,ly.Rate)
	}
	return out, g, nil
}