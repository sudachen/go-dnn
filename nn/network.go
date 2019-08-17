package nn

import "github.com/sudachen/go-dnn/mx"

type Network struct {
	Graph    *mx.Graph
	BatchLen int
}

func (n *Network) Release() {
	n.Graph.Release()
}

func getLoss(opts ...interface{}) Loss {
	for _, o := range opts {
		if loss, ok := o.(Loss); ok {
			return loss
		}
	}
	return nil
}

func Bind(ctx mx.Context, nb Block, input mx.Dimension, opts ...interface{}) (*Network, error) {
	sym, err := nb.Combine(mx.Input())
	if err != nil {
		return nil, err
	}
	loss := getLoss(opts...)
	g, err := mx.Compose(ctx, sym, loss, input, mx.Float32)
	if err != nil {
		return nil, err
	}
	f := &Network{Graph: g, BatchLen: input.Shape[0]}
	return f, nil
}

func (f *Network) Predict(data interface{}) ([][]float32, error) {
	if err := f.Graph.Input.SetValues(data); err != nil {
		return nil, err
	}
	if err := f.Graph.Forward(false); err != nil {
		return nil, err
	}
	o := f.Graph.Output.ValuesF32()
	r := make([][]float32, f.BatchLen)
	stride := len(o) / f.BatchLen
	for i := 0; i < f.BatchLen; i++ {
		r[i] = o[i*stride : (i+1)*stride]
	}
	return r, nil
}

func (f *Network) Train(data interface{}, label interface{}) error {
	return nil
}
