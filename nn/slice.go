package nn

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
)

type Slice struct {
	Axis int
	Begin int
	End int
	Name string
	Output     bool
	TurnOff    bool
}

func (ly *Slice) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	if ly.TurnOff {
		return in, g, nil
	}

	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("Slice%02d", mx.NextSymbolId())
	}
	out := mx.Slice(in,ly.Axis,ly.Begin,ly.End)
	out.SetName(ns)
	out.SetOutput(ly.Output)
	return out, g, nil
}

