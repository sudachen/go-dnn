package mx

import "io"

type Function struct {
	ctx    Context
	Input  *NDArray
	Output *NDArray
}

func (f *Function) Release() {
}

func (ctx Context) Bind(input Dimension, op *Symbol) (*Function, error) {
	f := &Function{ctx: ctx}
	return f, nil
}

func (ctx Context) BindBatch(input Dimension, batchLen int, op *Symbol, loss *Symbol) (*Function, error) {
	f := &Function{ctx: ctx}
	return f, nil
}

func (f *Function) LoadParams(reader io.Reader) error {
	return nil
}

func (f *Function) SaveParams(writer io.Writer) error {
	return nil
}

func (f *Function) InitParams() error {
	return nil
}

func (f *Function) Forward(train bool) error {
	return nil
}
