package nn

import (
	"github.com/sudachen/go-dnn/mx"
	"math"
)

type Adam struct {
	Loss mx.Loss

	Lr, Beta1, Beta2, Epsilon float64
}

type stAdam struct {
	Var   *mx.NDArray
	Mean  *mx.NDArray
	Index int
}

type implAdam struct {
	Adam
	States  map[*mx.NDArray]stAdam
	Upcount []float64
}

func (opt *Adam) Init() (Optimizer, error) {
	r := &implAdam{Adam: *opt, States: make(map[*mx.NDArray]stAdam)}
	if r.Loss == nil {
		r.Loss = L1Loss
	}
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
		st = stAdam{Var: v, Mean: m, Index: len(opt.Upcount)}
		opt.States[params] = st
		opt.Upcount = append(opt.Upcount, 0)
	}
	opt.Upcount[st.Index]++
	t := opt.Upcount[st.Index]
	coef1 := 1. - math.Pow(opt.Beta1, t)
	coef2 := 1. - math.Pow(opt.Beta2, t)
	lr := float32(opt.Lr * math.Sqrt(coef2) / coef1)
	beta1 := float32(opt.Beta1)
	beta2 := float32(opt.Beta2)
	epsilon := float32(opt.Epsilon)
	return mx.AdamUpdate(params, grads, st.Mean, st.Var, lr, beta1, beta2, epsilon, 0)
}

func (opt *implAdam) GetLoss() mx.Loss {
	return opt.Loss
}
