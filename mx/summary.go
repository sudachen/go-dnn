package mx

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/google/logger"
	"github.com/sudachen/go-dnn/mx/capi"
	"strings"
)

func (identity GraphIdentity) String() string {
	return hex.EncodeToString(identity[:])
}

func (g *Graph) Identity() GraphIdentity {
	if g.identity == nil {
		js, err := g.ToJson(false)
		if err != nil {
			panic(err.Error())
		}
		g.identity = &GraphIdentity{}
		h := sha1.New()
		h.Write(js)
		bs := h.Sum(nil)
		copy(g.identity[:], bs)
	}
	return *g.identity
}

func (g *Graph) ToJson(withLoss bool) ([]byte, error) {
	out := g.symLast
	if withLoss {
		out = g.symOut
	}
	return capi.ToJson(out)
}

type GraphJs struct {
	Nodes []struct {
		Op     string
		Name   string
		Attrs  map[string]string
		Inputs []interface{}
	}
}

type SummryArg struct {
	No   int
	Name string
}

type SummaryRow struct {
	No        int
	Name      string
	Operation string
	Params    int
	Dim       Dimension
	Args      []SummryArg
}

type Summary []*SummaryRow

func (g *Graph) Summary(withLoss bool) (Summary, error) {
	var (
		js     []byte
		err    error
		gjs    GraphJs
		shapes map[string][]int
	)
	if shapes, err = g.GetShapes(withLoss); err != nil {
		return nil, err
	}
	if js, err = g.ToJson(withLoss); err != nil {
		return nil, err
	}
	if err = json.Unmarshal(js, &gjs); err != nil {
		return nil, err
	}

	ns := map[string]*SummaryRow{}

	for lyno, ly := range gjs.Nodes {
		if ly.Op != "null" || lyno == 0 {
			n := &SummaryRow{No: len(ns), Name: ly.Name, Operation: ly.Op}
			if len(ly.Inputs) > 0 {
				for _, v := range ly.Inputs {
					inp := v.([]interface{})
					ly2 := gjs.Nodes[int(inp[0].(float64))]
					if ly2.Op == "null" {
						if ly2.Name == "_input" {
							//n.Params += g.Input.Dim().Total()
						} else if ly2.Name == "_label" {
							//n.Params += g.Label.Dim().Total()
						} else if p, ok := g.Params[ly2.Name]; ok {
							n.Params += p.Data.Dim().Total()
						}
					} else {
						n.Args = append(n.Args, SummryArg{ns[ly2.Name].No, ly2.Name})
					}
				}
			}
			if ly.Op == "Activation" {
				n.Operation += "(" + ly.Attrs["act_type"] + ")"
			} else if ly.Op == "SoftmaxActivation" {
				n.Operation += "(" + ly.Attrs["mode"] + ")"
			} else if ly.Op == "Pooling" {
				n.Operation += "(" + ly.Attrs["pool_type"] + ")"
			} else if ly.Op == "Convolution" {
				n.Operation += "(" + ly.Attrs["kernel"] + "/" + ly.Attrs["pad"] + "/" + ly.Attrs["stride"] + ")"
			}

			if dim0, ok := shapes[ly.Name+"_output"]; ok {
				n.Dim = Dim(dim0...)
			} else if dim0, ok := shapes[ly.Name+"_loss"]; ok {
				n.Dim = Dim(dim0...)
			}

			if lyno == 0 {
				n.Dim = g.Input.Dim()
			}

			ns[ly.Name] = n
		}
	}

	r := make(Summary, len(ns))
	for _, v := range ns {
		r[v.No] = v
	}

	return r, nil
}

func (g *Graph) SummaryOut(withLoss bool, out func(string)) error {
	var (
		nameLen, opLen, parLen, dimLen int = 9, 9, 9, 9
		err                            error
		sry                            Summary
	)
	if sry, err = g.Summary(withLoss); err != nil {
		return err
	}
	for _, v := range sry {
		if nameLen < len(v.Name) {
			nameLen = len(v.Name)
		}
		if opLen < len(v.Operation) {
			opLen = len(v.Operation)
		}
		p := fmt.Sprintf("%d", v.Params)
		if parLen < len(p) {
			parLen = len(p)
		}
		p = v.Dim.String()
		if dimLen < len(p) {
			dimLen = len(p)
		}
	}
	fth := fmt.Sprintf("%%-%ds | %%-%ds | %%-%ds | %%%ds", nameLen, opLen, dimLen, parLen)
	ft := fmt.Sprintf("%%-%ds | %%-%ds | %%-%ds | %%%dd", nameLen, opLen, dimLen, parLen)
	npars := 0
	hdr := fmt.Sprintf(fth, "Symbol", "Operation", "Output", "Params #")
	out(hdr)
	out(strings.Repeat("-", len(hdr)))
	for _, v := range sry {
		out(fmt.Sprintf(ft, v.Name, v.Operation, v.Dim, v.Params))
		npars += v.Params
	}
	out(strings.Repeat("-", len(hdr)))
	out(fmt.Sprintf("Total params: %d", npars))
	return nil
}

func (g *Graph) LogSummary(withLoss bool) error {
	return g.SummaryOut(withLoss, func(s string) { logger.Info(s) })
}

func (g *Graph) PrintSummary(withLoss bool) error {
	return g.SummaryOut(withLoss, func(s string) { fmt.Println(s) })
}
