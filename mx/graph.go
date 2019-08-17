package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx/capi"
	"io"
	"runtime"
)

type Loss = func(*Symbol, *Symbol) *Symbol

type Graph struct {
	Ctx   Context
	Dtype Dtype

	Input   *NDArray // network input
	Output  *NDArray // referencing to executor outputs[0]
	Label   *NDArray // loss function label
	Params  map[string]*NDArray
	Grads   []*NDArray // gradients
	Vars    map[string]capi.SymbolHandle
	Symbols map[*Symbol]capi.SymbolHandle

	Autograd map[string]int // params required autograd (with index in Grads array)

	Last capi.SymbolHandle // terminal symbol of network
	Out  capi.SymbolHandle // terminal symbol to execute
	Exec capi.ExecutorHandle

	Initializers map[string]VarInitializer
}

func (g *Graph) Release() {
	g.Input.Release()
	g.Label.Release()
	capi.ReleaseExecutor(g.Exec)

	for _, v := range g.Grads {
		v.Release()
	}
	for _, v := range g.Params {
		v.Release()
	}
	for _, v := range g.Symbols {
		capi.ReleaseSymbol(v)
	}
	for _, v := range g.Vars {
		capi.ReleaseSymbol(v)
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

func (g *Graph) allocate(shapes map[string][]int) error {
	for n, s := range shapes {
		if _, ok := g.Params[n]; !ok && n != "_output" {
			a := g.Ctx.Array(g.Dtype, Dim(s...))
			if a.Err() != nil {
				return a.Err()
			}
			g.Params[n] = a
		}
	}
	return nil
}

func (g *Graph) bind(input Dimension) error {
	var (
		shapes map[string][]int
		err    error
		names  []string
		o      []capi.NDArrayHandle
		d      Dimension
	)
	x := map[string][]int{"_input": input.Shape[:input.Len]}
	if shapes, err = capi.InferShapes(g.Last, x); err != nil {
		return err
	}
	if g.Input = g.Ctx.Array(g.Dtype, input); g.Input.Err() != nil {
		return g.Input.Err()
	}
	d = Dim(shapes["_output"]...)
	if g.Last != g.Out {
		if g.Label = g.Ctx.Array(g.Dtype, d); g.Label.Err() != nil {
			return g.Label.Err()
		}
		x["_label"] = d.Shape[:d.Len]
		if shapes, err = capi.InferShapes(g.Out, x); err != nil {
			return err
		}
	}
	if names, err = capi.ListArguments(g.Out); err != nil {
		return err
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
			args[i] = g.Params[name].handle
			if g.Last != g.Out && g.Autograd[name] == 1 {
				a := g.Ctx.Array(g.Dtype, g.Params[name].Dim())
				if a.Err() != nil {
					return a.Err()
				}
				g.Grads = append(g.Grads, a)
				grads[i] = a.handle
				a.Zeros()
			}
		}
	}
	if g.Exec, err = capi.Bind(g.Out, g.Ctx.DevType(), g.Ctx.DevNo(), args, grads); err != nil {
		return err
	}
	if o, err = capi.GetOutputs(g.Exec); err != nil {
		return err
	}
	g.Output = &NDArray{handle: o[0], ctx: g.Ctx, dim: d, dtype: g.Dtype}
	return nil
}

func Compose(
	ctx Context,
	sym *Symbol,
	symloss Loss,
	input Dimension,
	dtype Dtype) (*Graph, error) {

	var err error

	g := &Graph{
		Ctx:          ctx,
		Dtype:        dtype,
		Params:       make(map[string]*NDArray),
		Symbols:      make(map[*Symbol]capi.SymbolHandle),
		Vars:         make(map[string]capi.SymbolHandle),
		Autograd:     make(map[string]int),
		Initializers: make(map[string]VarInitializer),
	}

	if _, err = g.compose(Var("_input", nil), ""); err != nil {
		g.Release()
		return nil, err
	}

	if g.Last, err = g.compose(sym, ""); err != nil {
		g.Release()
		return nil, err
	}

	if symloss != nil {
		loss := Group(
			MakeLoss(BlockGrad(sym)),
			MakeLoss(symloss(sym, Var("_label", nil))))
		if g.Out, err = g.compose(loss, ""); err != nil {
			g.Release()
			return nil, err
		}
	} else {
		g.Out = g.Last
	}

	if err = g.bind(input); err != nil {
		g.Release()
		return nil, err
	}

	runtime.SetFinalizer(g, func(g *Graph) { g.Release() })
	return g, nil
}

func selectOp(s *Symbol, op, sop, sopR capi.MxnetOp) (capi.MxnetOp, string) {
	l, r := s.args[0], s.args[1]

	if l != nil && l.op == ScalarOp {
		return sopR, l.value
	}

	if r != nil && r.op == ScalarOp {
		return sop, r.value
	}

	return op, ""
}

func mkCommonSymbol(s *Symbol) (capi.SymbolHandle, error) {
	var mxnetop capi.MxnetOp
	switch s.op {
	case MeanOp:
		mxnetop = capi.OpMean
	case AbsOp:
		mxnetop = capi.OpAbs
	case BlockGradOp:
		mxnetop = capi.OpBlockGrad
	//case GroupOp: // handled directly by Graph.compose
	case MakeLossOp:
		mxnetop = capi.OpMakeLoss
	default:
		return nil, fmt.Errorf("unexpected symbol")
	}
	h, err := capi.NewSymbol(mxnetop, s.attr)
	if err != nil {
		return nil, err
	}
	return h, nil
}

func mkBinarySymbol(s *Symbol) (capi.SymbolHandle, error) {
	var mxnetop capi.MxnetOp
	var scalar string
	switch s.op {
	case AddOp:
		mxnetop, scalar = selectOp(s, capi.OpAdd, capi.OpAddScalar, capi.OpAddScalar)
	case SubOp:
		mxnetop, scalar = selectOp(s, capi.OpSub, capi.OpSubScalar, capi.OpSubScalarR)
	case MulOp:
		mxnetop, scalar = selectOp(s, capi.OpMul, capi.OpMulScalar, capi.OpMulScalar)
	case DivOp:
		mxnetop, scalar = selectOp(s, capi.OpDiv, capi.OpDivScalar, capi.OpDivScalarR)
	default:
		return nil, fmt.Errorf("unexpected symbol")
	}
	var (
		h   capi.SymbolHandle
		err error
	)
	if scalar != "" {
		h, err = capi.NewSymbol(mxnetop, nil, capi.KeyScalar, scalar)
	} else {
		h, err = capi.NewSymbol(mxnetop, nil)
	}
	if err != nil {
		return nil, err
	}
	return h, nil
}

var opmap = map[SymbolOp]func(*Symbol) (capi.SymbolHandle, error){
	AddOp: mkBinarySymbol,
	SubOp: mkBinarySymbol,
	MulOp: mkBinarySymbol,
	DivOp: mkBinarySymbol,
}

func mkSymbol(s *Symbol) (capi.SymbolHandle, error) {
	if f, ok := opmap[s.op]; ok {
		return f(s)
	} else {
		return mkCommonSymbol(s)
	}
}

func (g *Graph) subcompose(s *Symbol, ns string) ([]capi.SymbolHandle, error) {
	var err error
	var h capi.SymbolHandle
	var a []capi.SymbolHandle

	for _, v := range s.args {
		if h, err = g.compose(v, ns); err != nil {
			return nil, err
		}
		if h != nil {
			a = append(a, h)
		}
	}

	return a, nil
}

func (g *Graph) compose(s *Symbol, ns string) (capi.SymbolHandle, error) {

	if h, ok := g.Symbols[s]; ok {
		return h, nil
	}

	switch s.op {
	case InputOp:
		return g.Vars["_input"], nil
	case ScalarOp:
		return nil, nil
	case VarOp, AgVarOp:
		n := s.value
		if len(n) > 1 && n[0] != '_' {
			n = ns + n
		}
		if v, ok := g.Vars[n]; ok {
			return v, nil
		}
		h, e := capi.CreateVariable(n)
		if e != 0 {
			return nil, fmt.Errorf("failed to create mxnet variable")
		}
		g.Vars[n] = h
		if s.init != nil {
			g.Initializers[n] = s.init
		}
		if s.op == AgVarOp {
			g.Autograd[n] = 1 // write gradient ( other options 'null':0, 'add':2 )
		}
		return h, nil
	case NsOp:
		n := s.value
		if len(n) == 0 {
			return nil, fmt.Errorf("empty string in namespace name")
		}
		n = ns + n + "_"
		if n[0] != '_' {
			n = "_" + n
		}
		return g.compose(s.args[0], n)
	}

	var err error
	var op capi.SymbolHandle
	var a []capi.SymbolHandle

	a, err = g.subcompose(s, ns)
	if err != nil {
		return nil, err
	}

	if s.op == GroupOp {

		if op, err = capi.GroupSymbols(a); err != nil {
			return nil, err
		}

		g.Symbols[s] = op

	} else {

		if op, err = mkSymbol(s); err != nil {
			return nil, err
		}

		g.Symbols[s] = op

		name := fmt.Sprintf("%ssym%d", ns, len(g.Symbols))

		if err := capi.ComposeSymbol(op, name, a...); err != nil {
			return nil, err
		}
	}

	return op, nil
}

func (g *Graph) Forward(train bool) error {
	return capi.Forward(g.Exec, train)
}
