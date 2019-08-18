package nn

import "github.com/sudachen/go-dnn/mx"

func L1Loss(out *mx.Symbol, label *mx.Symbol) *mx.Symbol {
	loss := mx.Abs(mx.Sub(out, label))
	return mx.Mean(loss)
}

func L0Loss(out *mx.Symbol, label *mx.Symbol) *mx.Symbol {
	return out
}
