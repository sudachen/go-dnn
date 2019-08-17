package nn

import "github.com/sudachen/go-dnn/mx"

type Block interface {
	Combine(input *mx.Symbol) (*mx.Symbol, error)
}
