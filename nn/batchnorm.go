package nn

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
)

type BatchNorm struct {
	Name           string
	Mom, Epsilon   float32
	UseGlobalStats bool
}

func (ly *BatchNorm) Combine(in *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	ns := ly.Name
	if ns == "" {
		ns = fmt.Sprintf("BatchNorm%02d", mx.NextSymbolId())
	} else {
		ns += "$BN"
	}

	gamma := mx.Var(ns+"_gamma", &Const{1})
	beta := mx.Var(ns+"_beta", &Const{0})
	running_mean := mx.Var(ns+"_rmean", mx.Nograd, &Const{0})
	running_var := mx.Var(ns+"_rvar", mx.Nograd, &Const{1})
	out := mx.BatchNorm(in, gamma, beta, running_mean, running_var, ly.Mom, ly.Epsilon, ly.UseGlobalStats)
	out.SetName(ns)
	return out, g, nil
}

