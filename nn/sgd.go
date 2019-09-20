package nn

import "github.com/sudachen/go-dnn/mx"

type SGD struct {
	Lr, Mom float32

	LrMap map[int]float32
}

func (opt *SGD) Init(e int) (Optimizer, error) {
	r := &implSGD{SGD: *opt, States: make(map[*mx.NDArray]*mx.NDArray)}
	if r.Lr == 0 {
		r.Lr = locateLr(e, opt.LrMap, 0.01)
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
	if opt.Mom != 0 {
		st, ok := opt.States[params]
		if !ok {
			if st = params.NewLikeThis().Zeros(); st.Err() != nil {
				return st.Err()
			}
			opt.States[params] = st
		}
		return mx.SgdMomUpdate(params, grads, st, opt.Lr, opt.Mom, 0)
	}
	return mx.SgdUpdate(params, grads, opt.Lr, 0)
}
