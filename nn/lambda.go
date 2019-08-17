package nn

import "github.com/sudachen/go-dnn/mx"

type Lambda struct {
	F func(*mx.Symbol) *mx.Symbol
}

func (nb *Lambda) Combine(input *mx.Symbol) (*mx.Symbol, error) {
	return nb.F(input), nil
}
