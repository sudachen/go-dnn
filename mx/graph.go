package mx

import (
	"fmt"
	"github.com/sudachen/errors"
	"github.com/sudachen/go-dnn/fu"
	"github.com/sudachen/go-dnn/mx/capi"
	"runtime"
	"strings"
)

type GraphIdentity [20]byte // SHA1

type Loss interface {
	// out => loss
	Loss(*Symbol) *Symbol
}

type Param struct {
	Data     *NDArray
	Grad     *NDArray
	Shape    Dimension
	Autograd bool
}

type Graph struct {
	Ctx   Context
	Dtype Dtype

	Input   *NDArray  // network input referencing to Params["_input"]
	Output  *NDArray  // referencing to Outputs["_output_output"]
	Loss    *NDArray  // referencing to Outputs["_loss_loss"]
	Label   *NDArray  // loss function label referencing to Params["_label"]

	Outputs  map[string]*NDArray  // referencing to executor outputs except loss
	Params   map[string]*NDArray  // network parameters
	Shapes   map[string]Dimension // predefined param shape
	Autograd map[string]bool      // if param can be trained
	Grads    map[string]*NDArray  // training gradients

	Exec         capi.ExecutorHandle
	Initializers map[string]Inite
	Initialized  bool

	symOut, symLast capi.SymbolHandle

	vars    map[string]capi.SymbolHandle
	symbols map[*Symbol]capi.SymbolHandle
	auxs    []capi.NDArrayHandle

	identity *GraphIdentity
	alias    map[*Symbol]*Symbol
	outputs  map[string]*Symbol
	refs     map[string]capi.SymbolHandle
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
	g.alias = nil
	g.refs = nil
	g.outputs = nil
}

func (g *Graph) Release() {
	runtime.SetFinalizer(g, nil)

	g.symRelease()
	if g.symLast != g.symOut {
		capi.ReleaseSymbol(g.symLast)
		g.symLast = nil
	}
	capi.ReleaseSymbol(g.symOut)
	g.symOut = nil

	capi.ReleaseExecutor(g.Exec)
	g.Exec = nil

	for _, v := range g.Params {
		v.Release()
	}
	g.Params = nil
	for _, v := range g.Grads {
		v.Release()
	}
	g.Grads = nil

	for _, v := range g.auxs {
		capi.ReleaseNDArry(v)
	}
}

func (g *Graph) allocate(shapes map[string][]int) error {

	for n, s := range shapes {
		_, ok := g.Params[n]
		if !ok {
			if s2, ok := g.Shapes[n]; ok { s = s2.Slice() }
			a := g.Ctx.Array(g.Dtype, Dim(s...))
			if a.Err() != nil {
				return a.Err()
			}
			g.Params[n] = a
		}
	}

	return nil
}

func (g *Graph) GetShapes(withLoss bool) (map[string][]int, error) {
	var (
		err    error
		inter  capi.SymbolHandle
		sym    capi.SymbolHandle = g.symLast
		shapes map[string][]int
	)

	if withLoss {
		sym = g.symOut
	}

	if inter, err = capi.GetInternals(sym); err != nil {
		return nil, err
	}

	x := map[string][]int{"_input": g.Input.Dim().Slice()}
	n, err := capi.ListNames(sym, capi.ArgumentsNames)

	for _, name := range n {
		if p, ok := g.Shapes[name]; ok && p.Len != 0 {
			x[name] = p.Slice()
		}
	}

	if shapes, err = capi.InferShapes(inter, x, capi.WithArguments|capi.WithOutputs); err != nil {
		return nil, err
	}

	return shapes, nil
}

func (g *Graph) bind() error {
	var (
		shapes map[string][]int
		err    error
		names  []string
		o      []capi.NDArrayInfo
	)
	input := g.Input.Dim()
	x := map[string][]int{"_input": input.Shape[:input.Len]}

	if names, err = capi.ListNames(g.symOut, capi.ArgumentsNames); err != nil {
		return err
	}

	for _, n := range names {
		if p, ok := g.Shapes[n]; ok && p.Len != 0 {
			x[n] = p.Slice()
		}
	}

	shapes, err = capi.InferShapes(g.symOut, x, capi.WithArguments|capi.WithAuxStates|capi.WithoutOutput)
	if err != nil {
		return err
	}

	if err = g.allocate(shapes); err != nil {
		return err
	}

	args := make([]capi.NDArrayHandle, len(names))
	grads := make([]capi.NDArrayHandle, len(names))

	g.Input  = g.Params["_input"]
	g.Label  = g.Params["_label"]

	for i, name := range names {
		p := g.Params[name]
		if p != nil {
			args[i] = p.handle
			if g.symLast != g.symOut && g.Autograd[name] {
				a := g.Ctx.Array(g.Dtype,p.Dim())
				if a.Err() != nil {
					return a.Err()
				}
				g.Grads[name] = a
				grads[i] = a.handle
			}
		}
	}

	auxnam, _ := capi.ListNames(g.symOut, capi.AuxNames)
	aux := make([]capi.NDArrayHandle, len(auxnam))
	for i, name := range auxnam {
		if p, ok := g.Params[name]; ok {
			aux[i] = p.handle
		}
	}

	if g.Exec, err = capi.Bind(g.symOut, g.Ctx.DevType(), g.Ctx.DevNo(), args, grads, aux); err != nil {
		return err
	}

	if o, err = capi.GetOutputs(g.Exec); err != nil {
		return err
	}

	if names, err = capi.ListNames(g.symOut, capi.OutputNames); err != nil {
		return err
	}

	g.Outputs = make(map[string]*NDArray)
	for i, n := range names {
		v := o[i]
		if strings.HasSuffix(n,"_output") {
			n = strings.TrimSuffix(n, "_output")
		} else {
			n = strings.TrimSuffix(n, "_loss")
		}
		g.Outputs[n] = &NDArray{handle: v.Handle, ctx: g.Ctx, dim: Dim(v.Dim...), dtype: Dtype(v.Type)}
	}
	if g.symLast != g.symOut {
		g.Loss = g.Outputs["_loss"]
	}
	g.Output = g.Outputs["_output"]
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
		Params:       make(map[string]*NDArray),
		Grads:        make(map[string]*NDArray),
		Autograd:     make(map[string]bool),
		Shapes:       make(map[string]Dimension),
		symbols:      make(map[*Symbol]capi.SymbolHandle),
		vars:         make(map[string]capi.SymbolHandle),
		refs:         make(map[string]capi.SymbolHandle),
		alias:        make(map[*Symbol]*Symbol),
		outputs:      make(map[string]*Symbol),
		Initializers: make(map[string]Inite),
	}

	if g.Input = ctx.Array(dtype, input); g.Input.Err() != nil {
		g.Release()
		return nil, g.Input.Err()
	}

	if _, err = g.compose(Var("_input")); err != nil {
		g.Release()
		return nil, err
	}

	//Out := MakeLoss(BlockGrad(sym))
	Out := BlockGrad(sym)
	Out.SetName("_output")
	if last, err = g.compose(Out); err != nil {
		g.Release()
		return nil, err
	}
	out = last

	if loss != nil {
		symloss := loss.Loss(sym)
		Loss := MakeLoss(symloss)
		Loss.SetName("_loss")
		_,_ = g.compose(symloss)
		others := fu.ValsOf(g.outputs).([]*Symbol)
		outs := append([]*Symbol{Out,Loss},others...)
		if out, err = g.compose(Group(outs...)); err != nil {
			g.Release()
			return nil, err
		}
		if len(others) > 0 {
			outs := append([]*Symbol{Out},others...)
			if last, err = g.compose(Group(outs...)); err != nil {
				g.Release()
				return nil, err
			}
		}
	} else if len(g.outputs) > 0 {
		others := fu.ValsOf(g.outputs).([]*Symbol)
		outs := append([]*Symbol{Out},others...)
		if last, err = g.compose(Group(outs...)); err != nil {
			g.Release()
			return nil, err
		}
		out = last
	}

	g.symLast = last
	g.symOut = out
	g.symRelease() // other symbols are not necessary more

	if err = g.bind(); err != nil {
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

	if a, ok := g.alias[s]; ok {
		return g.symbols[a], nil
	}
	if h, ok := g.symbols[s]; ok {
		return h, nil
	}

	switch s.op {
	case OpInput_:
		return g.vars["_input"], nil
	case OpScalar_:
		return nil, nil
	case OpRef_:
		if s, ok := g.refs[s.name]; ok {
			return s, nil
		}
		return nil, errors.Errorf("symbol %s does not exist", s.name)
	case OpVar_, OpNogVar_:
		n := s.name
		if v, ok := g.vars[n]; ok {
			return v, nil
		}
		h, e := capi.CreateVariable(n)
		if e != 0 {
			return nil, errors.Errorf("failed to create mxnet variable")
		}
		g.vars[n] = h
		g.refs[n] = h
		if s.init != nil {
			g.Initializers[n] = s.init
		}
		if s.op != OpNogVar_ && n[0] != '_' {
			g.Autograd[n] = true
		}
		if s.dim.Len > 0 {
			g.Shapes[n] = s.dim.Like(g.Input.Dim())
		}
		return h, nil
	case OpOutput_:
		n := "*"+s.name
		if _,ok := g.outputs[n]; !ok {
			g.outputs[n] = BlockGrad(s.args[0]).SetName(n)
		}
		return g.compose(s.args[0])
	case OpBound_:
		h, err := g.compose(s.args[0])
		if err != nil {
			return nil, err
		}
		for _, v := range s.args[1:] {
			if _, err = g.compose(v); err != nil {
				return nil, err
			}
		}
		return h, nil
	case OpDepend_:
		for _, v := range s.args[1:] {
			if _, err := g.compose(v); err != nil {
				return nil, err
			}
		}
		return g.compose(s.args[0])
	case capi.OpZeros, capi.OpOnes, capi.OpRandomUniform:
		s1 := *s
		s1.attr = make(map[capi.MxnetKey]string)
		for key, value := range s.attr {
			s1.attr[key] = value
		}
		s1.attr[capi.KeyShape] = s.dim.Like(g.Input.Dim()).String()
		a := &s1
		g.alias[s] = a
		s = a
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
			name = fmt.Sprintf("%s@%s%02d", s.op.Value(), "sym", NextSymbolId())
		}

		if err := capi.ComposeSymbol(op, name, a...); err != nil {
			return nil, err
		}

		if s.name != "" {
			g.refs[s.name] = op
		}

		if s.output {
			n := "*"+name
			if _,ok := g.outputs[n]; !ok {
				g.outputs[n] = BlockGrad(s).SetName(n)
			}
		}
	}

	return op, nil
}

func (g *Graph) InitParam(name string) error {
	param := g.Params[name]
	if i, ok := g.Initializers[name]; ok && i != nil {
		if err := i.Inite(param); err != nil {
			return err
		}
	} else {
		var err error
		if name[0] == '_' {
			err = param.Zeros().Err()
		} else if strings.Index(name, "_bias") >= 0 {
			err = param.Zeros().Err()
		} else {
			err = param.Xavier(false, 2, 3).Err()
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (g *Graph) Initialize(inite func(*NDArray, string) error) error {
	keys := fu.SortedDictKeys(g.Params)
	for _, name := range keys {
		if inite != nil {
			param := g.Params[name]
			if err := inite(param, name); err != nil {
				return err
			}
		} else {
			if err := g.InitParam(name); err != nil {
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
