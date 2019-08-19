package mx

import (
	"github.com/sudachen/go-dnn/mx/capi"
	"math"
)

func (a *NDArray) Uniform(low float32, high float32) *NDArray {
	err := capi.ImperativeInvokeInplace1(
		capi.OpRandomUniform,
		a.handle,
		capi.KeyLow, low,
		capi.KeyHigh, high)
	a.err = err
	return a
}

func (a *NDArray) Normal(mean float32, scale float32) *NDArray {
	err := capi.ImperativeInvokeInplace1(
		capi.OpRandomNormal,
		a.handle,
		capi.KeyLoc, mean,
		capi.KeyScale, scale)
	a.err = err
	return a
}

func (a *NDArray) Zeros() *NDArray {
	err := capi.ImperativeInvokeInplace1(
		capi.OpZeros,
		a.handle)
	a.err = err
	return a
}

func (a *NDArray) Ones() *NDArray {
	return a.Fill(1)
}

func (a *NDArray) Fill(value float32) *NDArray {
	var err error
	if err = capi.ImperativeInvokeInplace1(
		capi.OpZeros,
		a.handle); err == nil {
		err = capi.ImperativeInvokeInOut1(
			capi.OpAddScalar,
			a.handle,
			a.handle,
			capi.KeyScalar,value);
	}
	a.err = err
	return a
}

func (a *NDArray) Xavier(gaussian bool, factor int, magnitude float32) *NDArray {
	d := a.Dim()
	hws := 1.
	scale := 1.
	if d.Len < 2 { factor = 1 }
	for i:=2; i<d.Len; i++ { hws *= float64(d.Shape[i]) }
	switch factor {
	case 0: scale = float64(d.Shape[1])*hws
	case 1: scale = float64(d.Shape[0])*hws
	case 2: scale = (float64(d.Shape[1])*hws + float64(d.Shape[0])*hws)/2.0
	}
	scale = math.Sqrt(float64(magnitude)/scale)
	if gaussian {
		return a.Normal(0, float32(scale))
	} else {
		return a.Uniform(-float32(scale),float32(scale))
	}
}
