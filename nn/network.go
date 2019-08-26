package nn

import "github.com/sudachen/go-dnn/mx"

type Network struct {
	*mx.Graph

	BatchSize   int
	Initialized bool
}

func (n *Network) Release() {
	n.Graph.Release()
}

func Bind(ctx mx.Context, nb Block, input mx.Dimension, loss mx.Loss) (*Network, error) {
	var (
		err  error
		sym  *mx.Symbol
	)
	if sym, _, err = nb.Combine(mx.Input(), nil); err != nil {
		return nil, err
	}
	g, err := mx.Compose(ctx, sym, loss, input, mx.Float32)
	if err != nil {
		return nil, err
	}
	f := &Network{Graph: g, BatchSize: input.Shape[0]}
	return f, nil
}

func (f *Network) Predict1(data interface{}, out []float32) error {
	if err := f.Graph.Input.SetValues(data); err != nil {
		return err
	}
	if err := f.Graph.Forward(false); err != nil {
		return err
	}
	if err := f.Graph.Outputs[0].CopyValuesTo(out); err != nil {
		return err
	}
	return nil
}

func (f *Network) Predict(data interface{}) ([][]float32, error) {
	out := make([]float32, f.Graph.Outputs[0].Dim().Total())
	if err := f.Predict1(data, out); err != nil {
		return nil, err
	}
	r := make([][]float32, f.BatchSize)
	stride := len(out) / f.BatchSize
	for i := 0; i < f.BatchSize; i++ {
		r[i] = out[i*stride : (i+1)*stride]
	}
	return r, nil
}

type AccFunc = func(data, label []float32) (bool, error)

func (f *Network) Test(data, label []float32, accfunc AccFunc) (float64, error) {
	var acc float64 = 0
	out := make([]float32, f.Graph.Outputs[0].Dim().Total())
	if err := f.Predict1(data, out); err != nil {
		return 0, err
	}
	count := f.Graph.Outputs[0].Len(0)
	outw := len(out) / count
	labelw := len(label) / count
	for i := 0; i < count; i++ {
		ok, err := accfunc(out[outw*i:outw*(i+1)], label[labelw*i:labelw*(i+1)])
		if err != nil {
			return 0, err
		}
		if ok {
			acc += 1
		}
	}
	return acc / float64(count), nil
}

func (f *Network) Train(data interface{}, label interface{}, opt Optimizer) error {
	if err := f.Graph.Input.SetValues(data); err != nil {
		return err
	}
	if f.Graph.Label != nil {
		if err := f.Graph.Label.SetValues(label); err != nil {
			return err
		}
	}
	if err := f.Graph.Forward(true); err != nil {
		return err
	}
	if err := f.Graph.Backward(); err != nil {
		return err
	}
	return f.Update(opt)
}

func (f *Network) Update(opt Optimizer) error {
	for _, p := range f.Graph.Params {
		if p.Autograd && p.Grad != nil {
			err := opt.Update(p.Data, p.Grad)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
