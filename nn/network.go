package nn

import "github.com/sudachen/go-dnn/mx"

type Network struct {
	*mx.Graph
	BatchLen int
}

func (n *Network) Release() {
	n.Graph.Release()
}

func Bind(ctx mx.Context, nb Block, input mx.Dimension, opts ...interface{}) (*Network, error) {
	sym, _, err := nb.Combine(mx.Input())
	if err != nil {
		return nil, err
	}
	g, err := sym.Compose()
	if err != nil {
		return nil, err
	}
	if err = g.Bind(ctx, input); err != nil {
		g.Release()
		return nil, err
	}
	f := &Network{Graph: g, BatchLen: input.Shape[0]}
	return f, nil
}

func (f *Network) Predict() error {
	return nil
}

func (f *Network) Train() error {
	return nil
}
