package nn

import "github.com/sudachen/go-dnn/mx"

type L0Loss struct{}

func (*L0Loss) Loss(out *mx.Symbol, _ *mx.Symbol) (*mx.Symbol, bool) {
	return out, false
}

type L1Loss struct{}

func (*L1Loss) Loss(out *mx.Symbol, label *mx.Symbol) (*mx.Symbol, bool) {
	return mx.Mean(mx.Abs(mx.Sub(out, label))), false
}

type L2Loss struct{}

func (*L2Loss) Loss(out *mx.Symbol, label *mx.Symbol) (*mx.Symbol, bool) {
	return mx.Mean(mx.Square(mx.Sub(out, label))), false
}

type SoftmaxCrossEntropyLoss struct{}

func (*SoftmaxCrossEntropyLoss) Loss(out *mx.Symbol, label *mx.Symbol) (*mx.Symbol, bool) {
	return mx.SoftmaxCrossEntropy(out, label), true
}

type LabelCrossEntropyLoss struct {
	Logit bool
}

func (l *LabelCrossEntropyLoss) Loss(out *mx.Symbol, label *mx.Symbol) (*mx.Symbol, bool) {
	return mx.Mean(mx.Mul(mx.Log(mx.Pick(out, label)), -1)), true
}
