package nn

import "github.com/sudachen/go-dnn/mx"

type Network struct {
	Graph *mx.Graph

	BatchLen    int
	Initialized bool
	Optimizer
}

func (n *Network) Release() {
	n.Graph.Release()
	if n.Optimizer != nil {
		n.Optimizer.Release()
	}
}

func Bind(ctx mx.Context, nb Block, input mx.Dimension, opt OptimizerConf) (*Network, error) {
	var (
		err  error
		sym  *mx.Symbol
		opti Optimizer
		loss mx.Loss
	)
	if sym, _, err = nb.Combine(mx.Input(), nil); err != nil {
		return nil, err
	}
	if opt != nil {
		if opti, err = opt.Init(); err != nil {
			return nil, err
		}
		loss = opti.GetLoss()
	}
	g, err := mx.Compose(ctx, sym, loss, input, mx.Float32)
	if err != nil {
		return nil, err
	}
	f := &Network{Graph: g, BatchLen: input.Shape[0], Optimizer: opti}
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
	r := make([][]float32, f.BatchLen)
	stride := len(out) / f.BatchLen
	for i := 0; i < f.BatchLen; i++ {
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

func (f *Network) Train(data interface{}, label interface{}) error {
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
	return f.Update()
}

func (f *Network) Update() error {
	for _, p := range f.Graph.Params {
		if p.Autograd && p.Grad != nil {
			err := f.Optimizer.Update(p.Data, p.Grad)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
