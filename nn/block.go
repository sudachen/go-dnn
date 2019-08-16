package nn

import "github.com/sudachen/go-dnn/mx"

type Block interface {
	Combine(*mx.Symbol) (*mx.Symbol, []string, error)
	Initialize(name string, param *mx.NDArray) error
}
