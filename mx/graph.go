package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx/internal"
	"io"
	"runtime"
)

type Graph struct {
	Context
	Input   *NDArray
	Output  *NDArray
	Params  map[string]*NDArray
	Indexes map[string]int
	Symbols []internal.SymbolHandle
	Last    *internal.SymbolHandle
}

func (g *Graph) Release() {
	g.Input.Release()
	g.Output.Release()
	for _, v := range g.Params {
		v.Release()
	}
	for _, v := range g.Symbols {
		internal.ReleaseSymbol(v)
	}
}

func (g *Graph) Close() error {
	g.Release()
	return nil
}

func (g *Graph) LoadParams(reader io.Reader) error {
	return nil
}

func (g *Graph) SaveParams(writer io.Writer) error {
	return nil
}

func (g *Graph) Bind(ctx Context, input Dimension) error {
	shapes, err := internal.InferShapes(g.Last, map[string][]int{"_input": input.Shape[:input.Len]})
	if err != nil {
		return err
	}
	for n, s := range shapes {
		a := ctx.Array(Float32, Dim(s...))
		if a.Err() != nil {
			return a.Err()
		}
		g.Params[n] = a
	}
	g.Input = ctx.Array(Float32, input)
	if g.Input.Err() != nil {
		return g.Input.Err()
	}
	return nil
}

func (s *Symbol) Compose() (*Graph, error) {
	var err error

	g := &Graph{
		Indexes: make(map[string]int),
		Params:  make(map[string]*NDArray),
		Symbols: make([]internal.SymbolHandle, 0, 32),
	}

	if _, err = g.compose(Var("_input")); err != nil {
		g.Release()
		return nil, err
	}

	if g.Last, err = g.compose(s); err != nil {
		g.Release()
		return nil, err
	}

	runtime.SetFinalizer(g, func(g *Graph) { g.Release() })
	return g, nil
}

func selectOp(s *Symbol, op, sop, sopR internal.MxnetOp) (internal.MxnetOp, string) {
	l, r := s.args[0], s.args[1]

	if l != nil && l.op == ScalarOp {
		return sopR, l.value
	}

	if r != nil && r.op == ScalarOp {
		return sop, r.value
	}

	return op, ""
}

func mkBinarySymbol(s *Symbol) (internal.SymbolHandle, string, error) {
	var mxnetop internal.MxnetOp
	var scalar string
	switch s.op {
	case AddOp:
		mxnetop, scalar = selectOp(s, internal.OpAdd, internal.OpAddScalar, internal.OpAddScalar)
	case SubOp:
		mxnetop, scalar = selectOp(s, internal.OpSub, internal.OpSubScalar, internal.OpSubScalarR)
	case MulOp:
		mxnetop, scalar = selectOp(s, internal.OpMul, internal.OpMulScalar, internal.OpMulScalar)
	case DivOp:
		mxnetop, scalar = selectOp(s, internal.OpDiv, internal.OpDivScalar, internal.OpDivScalarR)
	default:
		return nil, "", fmt.Errorf("unexpected symbol")
	}
	var (
		h   internal.SymbolHandle
		err error
	)
	if scalar != "" {
		h, err = internal.NewSymbol(mxnetop, nil, internal.KeyScalar, scalar)
	} else {
		h, err = internal.NewSymbol(mxnetop, nil)
	}
	if err != nil {
		return nil, "", err
	}
	return h, mxnetop.Value(), nil
}

func mkSymbol(s *Symbol) (internal.SymbolHandle, string, error) {
	switch s.op {
	case AddOp, SubOp, MulOp, DivOp:
		return mkBinarySymbol(s)
	default:
		return nil, "", fmt.Errorf("unexpected symbol")
	}
}

func (g *Graph) compose(s *Symbol) (internal.SymbolHandle, error) {
	// g.Symbols[0] - always _input

	if s.op == InputOp {
		return g.Symbols[0], nil
	}

	if s.op == ScalarOp {
		return nil, nil
	}

	if s.op == VarOp {
		if i, ok := g.Indexes[s.value]; ok {
			return g.Symbols[i], nil
		}
		h, e := internal.CreateVariable(s.value)
		if e != 0 {
			return nil, fmt.Errorf("failed to create mxnet variable")
		}
		g.Symbols = append(g.Symbols, h)
		g.Indexes[s.value] = len(g.Symbols) - 1
		return h, nil
	}

	var err error
	var text string
	var op, h internal.SymbolHandle
	var a []internal.SymbolHandle

	if op, text, err = mkSymbol(s); err != nil {
		return nil, err
	}
	name := fmt.Sprintf("sym%d_%s", len(g.Symbols), text)
	g.Symbols = append(g.Symbols, op)

	for _, v := range s.args {
		if h, err = g.compose(v); err != nil {
			return nil, err
		}
		if h != nil {
			g.Symbols = append(g.Symbols, h)
			a = append(a, h)
		}
	}

	if err := internal.ComposeSymbol(op, name, a...); err != nil {
		return nil, err
	}

	return op, nil
}
