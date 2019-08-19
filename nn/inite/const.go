package inite

import "github.com/sudachen/go-dnn/mx"

type Const struct {
	Value float32
}

func (x *Const) Inite(a *mx.NDArray) error {
	if x.Value == 0 {
		return a.Zeros().Err()
	}
	return a.Fill(x.Value).Err()
}
