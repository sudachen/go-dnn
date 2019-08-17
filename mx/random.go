package mx

import (
	"github.com/sudachen/go-dnn/mx/internal"
)

func (a *NDArray) Uniform(low float32, high float32) *NDArray {
	err := internal.ImperativeInvokeInplace1(
		internal.OpRandomUniform,
		a.handle,
		internal.KeyLow, low,
		internal.KeyHigh, high)
	a.err = err
	return a
}

func (a *NDArray) Zeros() *NDArray {
	err := internal.ImperativeInvokeInplace1(
		internal.OpZeros,
		a.handle)
	a.err = err
	return a
}
