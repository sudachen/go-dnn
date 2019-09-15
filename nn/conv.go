package nn

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
)

type Convolution struct {
	Channels   int
	Kernel     mx.Dimension
	Stride     mx.Dimension
	Padding    mx.Dimension
	Activation func(*mx.Symbol) *mx.Symbol
	WeightInit mx.Inite // none by default
	BiasInit   mx.Inite // &nn.Const{0} by default
	NoBias     bool
	Groups     bool
	Name       string
}

func (ly *Convolution) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	var (
		out, weight, bias *mx.Symbol
	)
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("Convolution%02d", mx.NextSymbolId())
	}
	weight = mx.Var(ns+"_weight", ly.WeightInit)
	if !ly.NoBias {
		init := ly.BiasInit
		if init == nil {
			init = &Const{0}
		}
		bias = mx.Var(ns+"_bias", init)
	}
	out = mx.Conv(in, weight, bias, ly.Channels, ly.Kernel, ly.Stride, ly.Padding, ly.Groups)
	out.SetName(ns)
	if ly.Activation != nil {
		out = ly.Activation(out)
	}
	return out, g, nil
}

type MaxPool struct {
	Kernel  mx.Dimension
	Stride  mx.Dimension
	Padding mx.Dimension
	Ceil    bool
	Name    string
}

func (ly *MaxPool) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	return mx.Pool(in, ly.Kernel, ly.Stride, ly.Padding, ly.Ceil, true), g, nil
}

type AvgPool struct {
	Kernel  mx.Dimension
	Stride  mx.Dimension
	Padding mx.Dimension
	Ceil    bool
}

func (ly *AvgPool) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	return mx.Pool(in, ly.Kernel, ly.Stride, ly.Padding, ly.Ceil, false), g, nil
}
