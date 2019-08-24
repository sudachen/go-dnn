package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/fu"
	"github.com/sudachen/go-dnn/mx/capi"
	"io"
	"runtime"
	"strings"
)

type GraphIdentity [20]byte // SHA1

type Loss interface {
	// out, label => loss, sparse
	// sparse means label dimensions reduced by last one
	Loss(*Symbol, *Symbol) (*Symbol, bool)
}

type Param struct {
	Data     *NDArray
	Grad     *NDArray
	Autograd bool
}

type Graph struct {
	Ctx   Context
	Dtype Dtype

	Input   *NDArray   // network input
	Outputs []*NDArray // referencing to executor outputs except loss
	Loss    *NDArray   // referencing to last executor output
	Label   *NDArray   // loss function label
	Params  map[string]Param

	Exec         capi.ExecutorHandle
	Initializers map[string]Inite
	Initialized  bool

	symOut, symLast capi.SymbolHandle

	vars    map[string]capi.SymbolHandle
	symbols map[*Symbol]capi.SymbolHandle

	identity *GraphIdentity
}

func (g *Graph) symRelease() {
	for _, v := range g.symbols {
		if v != g.symOut && v != g.symLast {
			capi.ReleaseSymbol(v)
		}
	}
	g.symbols = nil
	for _, v := range g.vars {
		capi.ReleaseSymbol(v)
	}
	g.vars = nil
}

func (g *Graph) Release() {
	g.symRelease()
	if g.symLast != g.symOut {
		capi.ReleaseSymbol(g.symLast)
	}
	capi.ReleaseSymbol(g.symOut)

	g.Input.Release()
	g.Label.Release()

	capi.ReleaseExecutor(g.Exec)
	g.Exec = nil

	for _, v := range g.Params {
		v.Data.Release()
		v.Grad.Release()
	}
	g.Params = nil
}

func (g *Graph) LoadParams(reader io.Reader) error {
	return nil
}

func (g *Graph) SaveParams(writer io.Writer) error {
	return nil
}

func (g *Graph) allocate(shapes map[string][]int) error {
	for n, s := range shapes {
		if n == "_label" {
			if g.Label = g.Ctx.Array(g.Dtype, Dim(s...)); g.Label.Err() != nil {
				return g.Label.Err()
			}
		} else if n != "_output" && n != "_input" {
			p, ok := g.Params[n]
			if !ok || p.Data == nil {
				a := g.Ctx.Array(g.Dtype, Dim(s...))
				if a.Err() != nil {
					return a.Err()
				}
				p.Data = a
				g.Params[n] = p
			}
		}
	}
	return nil
}

func (g *Graph) GetShapes(withLoss bool) (map[string][]int, error) {
	var (
		err                  error
		inter                capi.SymbolHandle
		sym                  capi.SymbolHandle = g.symLast
		input, label, output Dimension
		shapes               map[string][]int
	)

	if withLoss {
		sym = g.symOut
	}

	if inter, err = capi.GetInternals(sym); err != nil {
		return nil, err
	}

	input = g.Input.Dim()
	x := map[string][]int{"_input": input.Shape[:input.Len]}
	if withLoss && g.Label != nil {
		label = g.Label.Dim()
		x["_label"] = label.Shape[:label.Len]
	}

	if shapes, err = capi.InferShapes(inter, x, true); err != nil {
		return nil, err
	}

	shapes["_input"] = input.Shape[1:input.Len]
	if label.Len != 0 {
		shapes["_label"] = label.Shape[:label.Len]
	}

	for n, o := range g.Outputs {
		output = o.Dim()
		shapes[fmt.Sprintf("_output%d", n)] = output.Shape[:label.Len]
	}

	return shapes, nil
}

func (g *Graph) bind(input Dimension, sparse bool) error {
	var (
		shapes map[string][]int
		err    error
		names  []string
		o      []capi.NDArrayInfo
		d      Dimension
	)
	x := map[string][]int{"_input": input.Shape[:input.Len]}
	if shapes, err = capi.InferShapes(g.symLast, x, false); err != nil {
		return err
	}
	if g.Input = g.Ctx.Array(g.Dtype, input); g.Input.Err() != nil {
		return g.Input.Err()
	}
	if names, err = capi.ListNames(g.symOut, false); err != nil {
		return err
	}
	if g.symLast != g.symOut && fu.Contains(names, "_label") {
		d = Dim(shapes["_output"]...)
		if sparse {
			d.Len -= 1
		}
		x["_label"] = d.Shape[:d.Len]
		if shapes, err = capi.InferShapes(g.symOut, x, false); err != nil {
			return err
		}
	}

	if err = g.allocate(shapes); err != nil {
		return err
	}

	args := make([]capi.NDArrayHandle, len(names))
	grads := make([]capi.NDArrayHandle, len(names))
	for i, name := range names {
		if name == "_input" {
			args[i] = g.Input.handle
		} else if name == "_label" {
			args[i] = g.Label.handle
		} else {
			p := g.Params[name]
			args[i] = p.Data.handle
			if g.symLast != g.symOut && p.Autograd {
				a := g.Ctx.Array(g.Dtype, p.Data.Dim())
				if a.Err() != nil {
					return a.Err()
				}
				p.Grad = a
				grads[i] = a.handle
				g.Params[name] = p
			}
		}
	}

	if g.Exec, err = capi.Bind(g.symOut, g.Ctx.DevType(), g.Ctx.DevNo(), args, grads); err != nil {
		return err
	}

	if o, err = capi.GetOutputs(g.Exec); err != nil {
		return err
	}

	g.Outputs = make([]*NDArray, len(o))
	for i, v := range o {
		g.Outputs[i] = &NDArray{handle: v.Handle, ctx: g.Ctx, dim: Dim(v.Dim...), dtype: Dtype(v.Type)}
	}
	if g.symLast != g.symOut {
		g.Loss = g.Outputs[len(o)-1]
		g.Outputs = g.Outputs[:len(o)-1]
	} else {
		g.Loss = g.Outputs[0]
	}
	return nil
}

func Compose(
	ctx Context,
	sym *Symbol,
	loss Loss,
	input Dimension,
	dtype Dtype) (*Graph, error) {

	var (
		err  error
		last capi.SymbolHandle
		out  capi.SymbolHandle
	)

	g := &Graph{
		Ctx:          ctx,
		Dtype:        dtype,
		Params:       make(map[string]Param),
		symbols:      make(map[*Symbol]capi.SymbolHandle),
		vars:         make(map[string]capi.SymbolHandle),
		Initializers: make(map[string]Inite),
	}

	if _, err = g.compose(Var("_input")); err != nil {
		g.Release()
		return nil, err
	}

	if last, err = g.compose(sym); err != nil {
		g.Release()
		return nil, err
	}

	out = last

	var sparse bool
	if loss != nil {
		var symloss *Symbol
		symloss, sparse = loss.Loss(sym, Var("_label"))
		l := Group(
			MakeLoss(BlockGrad(sym)),
			MakeLoss(symloss))
		if out, err = g.compose(l); err != nil {
			g.Release()
			return nil, err
		}
	}

	g.symLast = last
	g.symOut = out
	g.symRelease() // other symbols are not necessary more

	if err = g.bind(input, sparse); err != nil {
		g.Release()
		return nil, err
	}

	runtime.SetFinalizer(g, func(g *Graph) { g.Release() })
	return g, nil
}

func (g *Graph) subcompose(s *Symbol) ([]capi.SymbolHandle, error) {
	var err error
	var h capi.SymbolHandle
	var a []capi.SymbolHandle

	for _, v := range s.args {
		if h, err = g.compose(v); err != nil {
			return nil, err
		}
		if h != nil {
			a = append(a, h)
		}
	}

	return a, nil
}

func (g *Graph) compose(s *Symbol) (capi.SymbolHandle, error) {

	if h, ok := g.symbols[s]; ok {
		return h, nil
	}

	switch s.op {
	case OpInput_:
		return g.vars["_input"], nil
	case OpScalar_:
		return nil, nil
	case OpVar_, OpNogVar_:
		n := s.name
		if v, ok := g.vars[n]; ok {
			return v, nil
		}
		h, e := capi.CreateVariable(n)
		if e != 0 {
			return nil, fmt.Errorf("failed to create mxnet variable")
		}
		g.vars[n] = h
		if s.init != nil {
			g.Initializers[n] = s.init
		}
		if s.op != OpNogVar_ && n != "_input" && n != "_label" {
			p := g.Params[n]
			p.Autograd = true
			g.Params[n] = p
		}
		return h, nil
	}

	var err error
	var op capi.SymbolHandle
	var a []capi.SymbolHandle

	a, err = g.subcompose(s)
	if err != nil {
		return nil, err
	}

	if s.op == OpGroup_ {

		if op, err = capi.GroupSymbols(a); err != nil {
			return nil, err
		}

		g.symbols[s] = op

	} else {

		if op, err = capi.NewSymbol(s.op, s.attr); err != nil {
			return nil, err
		}

		g.symbols[s] = op

		name := s.name
		if len(name) < 3 {
			name = fmt.Sprintf("%s%02d", "sym", NextSymbolId())
		}

		if err := capi.ComposeSymbol(op, name, a...); err != nil {
			return nil, err
		}
	}

	return op, nil
}

func (g *Graph) Initialize(inite func(*NDArray, string) error) error {
	keys := fu.SortedDictKeys(g.Params)
	for _, name := range keys {
		param := g.Params[name]
		if i, ok := g.Initializers[name]; ok && i != nil {
			if err := i.Inite(param.Data); err != nil {
				return err
			}
		} else if inite != nil {
			if err := inite(param.Data, name); err != nil {
				return err
			}
		} else {
			var err error
			if strings.Index(name, "_bias") >= 0 {
				err = param.Data.Zeros().Err()
			} else {
				err = param.Data.Xavier(false, 2, 3).Err()
			}
			if err != nil {
				return err
			}
		}
	}
	g.Initialized = true
	return nil
}

func (g *Graph) Forward(train bool) error {
	if !g.Initialized {
		err := g.Initialize(nil)
		if err != nil {
			return err
		}
	}
	return capi.Forward(g.Exec, train)
}

func (g *Graph) Backward() error {
	return capi.Backward(g.Exec)
}
