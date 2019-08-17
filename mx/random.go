package mx

import "github.com/sudachen/go-dnn/mx/capi"

func (a *NDArray) Uniform(low float32, high float32) *NDArray {
	err := capi.ImperativeInvokeInplace1(
		capi.OpRandomUniform,
		a.handle,
		capi.KeyLow, low,
		capi.KeyHigh, high)
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
