package nn

import (
	"github.com/sudachen/go-dnn/mx"
	//"math"
)

type Adam struct {
	Loss mx.Loss

	Lr, Beta1, Beta2, Epsilon float32
}

type stAdam struct {
	Var   *mx.NDArray
	Mean  *mx.NDArray
	Index int
}

type implAdam struct {
	Adam
	States map[*mx.NDArray]stAdam
}

func (opt *Adam) Init() (Optimizer, error) {
	r := &implAdam{Adam: *opt, States: make(map[*mx.NDArray]stAdam)}
	if r.Lr == 0 {
		r.Lr = 0.01
	}
	if r.Beta1 == 0 {
		r.Beta1 = 0.9
	}
	if r.Beta2 == 0 {
		r.Beta2 = 0.999
	}
	if r.Epsilon == 0 {
		r.Epsilon = 1e-8
	}
	return r, nil
}

func (opt *implAdam) Release() {
	for _, v := range opt.States {
		v.Var.Release()
		v.Mean.Release()
	}
}

func (opt *implAdam) Update(params *mx.NDArray, grads *mx.NDArray) error {
	st, ok := opt.States[params]
	if !ok {
		var v, m *mx.NDArray
		if v = params.NewLikeThis().Zeros(); v.Err() != nil {
			return v.Err()
		}
		if m = params.NewLikeThis().Zeros(); v.Err() != nil {
			v.Release()
			return v.Err()
		}
		st = stAdam{Var: v, Mean: m}
		opt.States[params] = st
	}
	return mx.AdamUpdate(params, grads, st.Mean, st.Var, opt.Lr, opt.Beta1, opt.Beta2, opt.Epsilon, 0)
}

func (opt *implAdam) GetLoss() mx.Loss {
	return opt.Loss
}
