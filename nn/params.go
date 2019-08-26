package nn

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"github.com/sudachen/go-dnn/mx"
	"io"
	"os"
)

type Params struct {
	P map[string][]float32 `json:"params"`
}

func ParamsOf(net *Network) (Params, error) {
	p := Params{map[string][]float32{}}
	for n, d := range net.Params {
		dm := d.Data.Dim()
		a := make([]float32, dm.Total()+5)
		a[0] = float32(dm.Len)
		for i := 0; i < 4; i++ {
			a[i+1] = float32(dm.Shape[i])
		}
		if err := d.Data.CopyValuesTo(a[5:]); err != nil {
			return Params{}, err
		}
		p.P[n] = a
	}
	return p, nil
}

func (p Params) check(net *Network, force bool) error {
	for n, d := range net.Params {
		a, ok := p.P[n]
		if ok {
			dm := d.Data.Dim()
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
		} else {
			if !force {
				return fmt.Errorf("absent parameter %v required by network", n)
			}
		}
	}
	return nil
}

func (p Params) Setup(net *Network, force bool) error {
	if err := p.check(net, force); err != nil {
		return err
	}
	for n, d := range net.Params {
		a, ok := p.P[n]
		if ok {
			if err := d.Data.SetValues(a[5:]); err != nil {
				return err
			}
		}
	}
	net.Graph.Initialized = true
	return nil
}

func LoadParams(fname string) (Params, error) {
	f, err := os.Open(fname)
	if err != nil {
		return Params{}, err
	}
	defer f.Close()
	p := Params{}
	if err = p.Read(f); err != nil {
		return Params{}, err
	}
	return p, nil
}

func (p *Params) Read(reader io.Reader) error {
	gz, err := gzip.NewReader(reader)
	defer gz.Close()
	if err != nil {
		return err
	}
	if err = json.NewDecoder(gz).Decode(p); err != nil {
		return err
	}
	return nil
}

func (p *Params) Save(fname string) error {
	f, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer f.Close()
	if err = p.Write(f); err != nil {
		return err
	}
	return nil
}

func (p Params) Write(writer io.Writer) error {
	gz, err := gzip.NewWriterLevel(writer, gzip.BestCompression)
	if err != nil {
		return err
	}
	if err = json.NewEncoder(gz).Encode(&p); err != nil {
		return err
	}
	return gz.Close()
}
