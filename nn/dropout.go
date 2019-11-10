package nn

import "github.com/sudachen/go-dnn/mx"

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

