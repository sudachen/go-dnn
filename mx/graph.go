package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/fu"
	"github.com/sudachen/go-dnn/mx/capi"
	"io"
	"runtime"
	"strings"
)

type Loss = func(*Symbol, *Symbol) *Symbol

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

	vars    map[string]capi.SymbolHandle
	symbols map[*Symbol]capi.SymbolHandle
}

func (g *Graph) symRelease() {
	for _, v := range g.symbols {
		capi.ReleaseSymbol(v)
	}
	g.symbols = nil
	for _, v := range g.vars {
		capi.ReleaseSymbol(v)
	}
	g.vars = nil
}

func (g *Graph) Release() {
	g.symRelease()
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
		if n != "_output" && n != "_label" && n != "_input" {
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

func (g *Graph) bind(input Dimension, out, last capi.SymbolHandle) error {
	var (
		shapes map[string][]int
		err    error
		names  []string
		o      []capi.NDArrayInfo
		d      Dimension
	)
	x := map[string][]int{"_input": input.Shape[:input.Len]}
	if shapes, err = capi.InferShapes(last, x); err != nil {
		return err
	}
	if g.Input = g.Ctx.Array(g.Dtype, input); g.Input.Err() != nil {
		return g.Input.Err()
	}
	if names, err = capi.ListArguments(out); err != nil {
		return err
	}
	if last != out && fu.Contains(names, "_label") {
		d = Dim(shapes["_output"]...)
		if g.Label = g.Ctx.Array(g.Dtype, d); g.Label.Err() != nil {
			return g.Label.Err()
		}
		x["_label"] = d.Shape[:d.Len]
		if shapes, err = capi.InferShapes(out, x); err != nil {
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
			if last != out && p.Autograd {
				a := g.Ctx.Array(g.Dtype, p.Data.Dim())
				if a.Err() != nil {
					return a.Err()
				}
				p.Grad = a
				grads[i] = a.handle
				a.Zeros()
				g.Params[name] = p
			}
		}
	}

	if g.Exec, err = capi.Bind(out, g.Ctx.DevType(), g.Ctx.DevNo(), args, grads); err != nil {
		return err
	}
	g.symRelease() // symbols are not necessary more

	if o, err = capi.GetOutputs(g.Exec); err != nil {
		return err
	}

	g.Outputs = make([]*NDArray, len(o))
	for i, v := range o {
		g.Outputs[i] = &NDArray{handle: v.Handle, ctx: g.Ctx, dim: Dim(v.Dim...), dtype: Dtype(v.Type)}
	}
	if last != out {
		g.Loss = g.Outputs[len(o)-1]
		g.Outputs = g.Outputs[:len(o)-1]
	}
	return nil
}

func Compose(
	ctx Context,
	sym *Symbol,
	symloss Loss,
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

	if _, err = g.compose(Var("_input"), ""); err != nil {
		g.Release()
		return nil, err
	}

	if last, err = g.compose(sym, ""); err != nil {
		g.Release()
		return nil, err
	}

	if symloss != nil {
		loss := Group(
			MakeLoss(BlockGrad(sym)),
			MakeLoss(symloss(sym, Var("_label"))))
		if out, err = g.compose(loss, ""); err != nil {
			g.Release()
			return nil, err
		}
	} else {
		out = last
	}

	if err = g.bind(input, out, last); err != nil {
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
	case PowOp:
		mxnetop, scalar = selectOp(s, capi.OpNoOp, capi.OpPowerScalar, capi.OpPowerScalarR)
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
	PowOp: mkBinarySymbol,
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

	if h, ok := g.symbols[s]; ok {
		return h, nil
	}

	switch s.op {
	case InputOp:
		return g.vars["_input"], nil
	case ScalarOp:
		return nil, nil
	case VarOp, AgVarOp:
		n := s.value
		if len(n) > 1 && n[0] != '_' {
			n = ns + n
		}
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
		if s.op == AgVarOp {
			p := g.Params[n]
			p.Autograd = true
			g.Params[n] = p
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

		g.symbols[s] = op

	} else {

		if op, err = mkSymbol(s); err != nil {
			return nil, err
		}

		g.symbols[s] = op

		name := fmt.Sprintf("%ssym%d", ns, len(g.symbols))

		if err := capi.ComposeSymbol(op, name, a...); err != nil {
			return nil, err
		}
	}

	return op, nil
}

func (g *Graph) Initialize(inite func(*NDArray,string)error) error{
	for name, param := range g.Params {
		if i, ok := g.Initializers[name]; ok && i != nil {
			if err := i.Inite(param.Data); err != nil {
				return err
			}
		} else {
			if err := inite(param.Data,name); err != nil {
				return err
			}

		}
	}
	g.Initialized = true
	return nil
}

func (g *Graph) Forward(train bool) error {
	if !g.Initialized {
		err := g.Initialize(func(a *NDArray, name string)error{
			if strings.Index(name,"_bias") >= 0 {
				return a.Zeros().Err()
			}
			return a.Xavier(false,2,3).Err()
		})
		if err != nil {
			return err
		}
	}
	return capi.Forward(g.Exec, train)
}

func (g *Graph) Backward() error {
	return capi.Backward(g.Exec)
}
