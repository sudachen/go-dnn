package nn

import "github.com/sudachen/go-dnn/mx"

type OptimizerConf interface {
	Init(epoch int) (Optimizer, error)
}

type Optimizer interface {
	Release()
	Update(params *mx.NDArray, grads *mx.NDArray) error
}

func locateLr(epoch int, lrmap map[int]float32, dflt float32) float32 {
	lr := dflt
	if lrmap != nil {
		e := -1
		for fromEpoch, lr2 := range lrmap {
			if fromEpoch > e && fromEpoch <= epoch {
				lr = lr2
			}
		}
	}
	return lr
}
