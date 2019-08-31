package nn

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
)

type Params struct {
	P map[string][]float32 `json:"params"`
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
