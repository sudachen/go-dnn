package nn

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
)

type Network struct {
	*mx.Graph

	BatchSize int
}

func (n *Network) Release() {
	n.Graph.Release()
}

func Bind(ctx mx.Context, nb Block, input mx.Dimension, loss mx.Loss) (*Network, error) {
	var (
		err error
		sym *mx.Symbol
	)
	mx.ResetSymbolId(0)
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
	if err := f.Graph.Output.CopyValuesTo(out); err != nil {
		return err
	}
	return nil
}

func (f *Network) Predict(data interface{}) ([][]float32, error) {
	out := make([]float32, f.Graph.Output.Dim().Total())
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

type Metric interface {
	Reset()
	Collect(data, label []float32)
	Value() float32
	Satisfy() bool
}

func (f *Network) Test(data, label []float32, metric Metric) (err error) {
	out := make([]float32, f.Graph.Output.Dim().Total())
	if err = f.Predict1(data, out); err != nil {
		return
	}
	count := f.Graph.Output.Len(0)
	outw := len(out) / count
	labelw := len(label) / count
	for i := 0; i < count; i++ {
		metric.Collect(out[outw*i:outw*(i+1)], label[labelw*i:labelw*(i+1)])
	}
	return
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
	for k, g := range f.Graph.Grads {
		err := opt.Update(f.Graph.Params[k], g)
		if err != nil {
			return err
		}
	}
	return nil
}

func (f *Network) LoadParamsFile(filename string, force bool) error {
	p, err := LoadParams(filename)
	if err != nil {
		return err
	}
	return f.SetParams(p, force)
}

func (f *Network) SaveParamsFile(filename string) error {
	p, err := f.GetParams()
	if err != nil {
		return err
	}
	return p.Save(filename)
}

func (f *Network) SetParams(p Params, force bool) error {
	if err := f.checkParams(p, force); err != nil {
		return err
	}
	return f.Graph.Initialize(func(d *mx.NDArray, n string) error {
		a, ok := p.P[n]
		if ok {
			return d.SetValues(a[5:])
		}
		return f.Graph.InitParam(n)
	})
}

func (f *Network) checkParams(p Params, force bool) error {
	for n, d := range f.Params {
		a, ok := p.P[n]
		if ok {
			dm := d.Dim()
			if dm.Total() == len(a)-5 {
				if !force {
					x := mx.Dimension{Len: int(a[0]), Shape: [4]int{int(a[1]), int(a[2]), int(a[3]), int(a[4])}}
					if dm != x {
						return fmt.Errorf("parameter %v has dim %v but network requires %v",
							n, x, dm)
					}
				}
			} else {
				return fmt.Errorf("parameter %v has %d values but network requires %d",
					n, len(a)-5, dm.Total())
			}
		} else if n[0] != '_' {
			if !force {
				return fmt.Errorf("absent parameter %v required by network", n)
			}
		} else {

		}
	}
	return nil
}

func (net *Network) GetParams() (Params, error) {
	p := Params{map[string][]float32{}}
	for n, d := range net.Params {
		if n[0] != '_' {
			dm := d.Dim()
			a := make([]float32, dm.Total()+5)
			a[0] = float32(dm.Len)
			for i := 0; i < 4; i++ {
				a[i+1] = float32(dm.Shape[i])
			}
			if err := d.ReCopyValuesTo(a[5:]); err != nil {
				return Params{}, err
			}
			p.P[n] = a
		}
	}
	return p, nil
}
