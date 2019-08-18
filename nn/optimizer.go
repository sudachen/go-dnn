package nn

import "github.com/sudachen/go-dnn/mx"

type OptimizerConf interface {
	Init() (Optimizer, error)
}

type Optimizer interface {
	Release()
	Update(params *mx.NDArray, grads *mx.NDArray) error
	GetLoss() mx.Loss
}

type SGD struct {
	Loss         mx.Loss
	LearningRate float32
	Momentum     float32
}

func (opt *SGD) Init() (Optimizer, error) {
	r := &implSGD{SGD: *opt, States: make(map[*mx.NDArray]*mx.NDArray)}
	if r.Loss == nil {
		r.Loss = L1Loss
	}
	if r.LearningRate == 0 {
		r.LearningRate = 0.01
	}
	if r.Momentum == 0 {
		r.Momentum = 0.8
	}
	return r, nil
}

type implSGD struct {
	SGD
	States map[*mx.NDArray]*mx.NDArray
}

func (opt *implSGD) Release() {
	for _, v := range opt.States {
		v.Release()
	}
}

func (opt *implSGD) Update(params *mx.NDArray, grads *mx.NDArray) error {
	st, ok := opt.States[params]
	if !ok {
		if st = params.NewLikeThis().Zeros(); st.Err() != nil {
			return st.Err()
		}
		opt.States[params] = st
	}
	return mx.SgdMomUpdate(params, grads, st, opt.LearningRate, opt.Momentum, 0)
}

func (opt *implSGD) GetLoss() mx.Loss {
	return opt.Loss
}

type Adam struct {
	Loss                  mx.Loss
	LearningRate          float32
	Beta1, Beta2, Epsilon float32
}
