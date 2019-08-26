package nn

import "github.com/sudachen/go-dnn/mx"

type OptimizerConf interface {
	Init() (Optimizer, error)
}

type Optimizer interface {
	Release()
	Update(params *mx.NDArray, grads *mx.NDArray) error
}
