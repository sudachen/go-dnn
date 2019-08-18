package nn

import "github.com/sudachen/go-dnn/mx"

func L0Loss(out *mx.Symbol, _ *mx.Symbol) *mx.Symbol {
	return out
}

func L1Loss(out *mx.Symbol, label *mx.Symbol) *mx.Symbol {
	return mx.Mean(mx.Abs(mx.Sub(out, label)))
}
