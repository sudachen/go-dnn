package nn

import "github.com/sudachen/go-dnn/mx"

func L1Loss(out *mx.Symbol, label *mx.Symbol, _ ...mx.Loss) *mx.Symbol {
	loss := mx.Abs(mx.Sub(out, label))
	return mx.Mean(loss)
}
