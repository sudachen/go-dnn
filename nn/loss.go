package nn

import (
	"github.com/sudachen/go-dnn/fu"
	"github.com/sudachen/go-dnn/mx"
)

type L0Loss struct{}

func (*L0Loss) Loss(out *mx.Symbol) *mx.Symbol {
	return out
}

type L1Loss struct{ Num int }

func (loss *L1Loss) Loss(out *mx.Symbol) *mx.Symbol {
	n := fu.IfElseI(loss.Num==0,1,loss.Num)
	label := mx.Var("_label", mx.Dim(0,n))
	return mx.Mean(mx.Abs(mx.Sub(out, label)))
}

type L2Loss struct{ Num int }

func (loss *L2Loss) Loss(out *mx.Symbol) *mx.Symbol {
	n := fu.IfElseI(loss.Num==0,1,loss.Num)
	label := mx.Var("_label", mx.Dim(0,n))
	return mx.Mean(mx.Square(mx.Sub(out, label)))
}

type SoftmaxCrossEntropyLoss struct{}

func (*SoftmaxCrossEntropyLoss) Loss(out *mx.Symbol) *mx.Symbol {
	label := mx.Var("_label", mx.Dim(0, 1))
	return mx.SoftmaxCrossEntropy(out, label)
}

type LabelCrossEntropyLoss struct {}

func (*LabelCrossEntropyLoss) Loss(out *mx.Symbol) *mx.Symbol {
	label := mx.Var("_label", mx.Dim(0, 1))
	return mx.Mean(mx.Mul(mx.Log(mx.Pick(out, label)), -1))
}

type CrossEntropyLoss struct { Num int }

func (loss *CrossEntropyLoss) Loss(out *mx.Symbol) *mx.Symbol {
	n := fu.IfElseI(loss.Num==0,1,loss.Num)
	label := mx.Var("_label", mx.Dim(0,n))
	return mx.Mean(mx.Mul(mx.Log(mx.Pick(out, label)), -1))
}
