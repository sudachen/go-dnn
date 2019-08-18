package nn

import "github.com/sudachen/go-dnn/mx"

type Network struct {
	Graph *mx.Graph

	BatchLen int
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
	if sym, err = nb.Combine(mx.Input()); err != nil {
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

func (f *Network) Predict(data interface{}) ([][]float32, error) {
	if err := f.Graph.Input.SetValues(data); err != nil {
		return nil, err
	}
	if err := f.Graph.Forward(false); err != nil {
		return nil, err
	}
	o := f.Graph.Outputs[0].ValuesF32()
	r := make([][]float32, f.BatchLen)
	stride := len(o) / f.BatchLen
	for i := 0; i < f.BatchLen; i++ {
		r[i] = o[i*stride : (i+1)*stride]
	}
	return r, nil
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
