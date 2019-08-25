package mx

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx/capi"
)

type Context int

const (
	NullContext Context = 0
	CPU         Context = 1
	GPU0        Context = 2
	GPU1        Context = 1002
)

func Gpu(no int) Context {
	if no >= 0 && no < GpuCount() {
		return Context(no*1000) + GPU0
	}
	return NullContext
}

func (c Context) DevType() int {
	return int(c) % 1000
}

func (c Context) DevNo() int {
	return int(c) / 1000
}

func (c Context) IsGPU() bool {
	return c.DevType() == 2
}

func (c Context) String() string {
	switch c.DevType() {
	case 0:
		return "NullContext"
	case 1:
		return "CPU"
	case 2:
		return fmt.Sprintf("GPU%d", c.DevNo())
	}
	return "InvalidContext"
}

func (c Context) RandomSeed(seed int) {
	_ = capi.ContextRandomSeed(seed, c.DevType(), c.DevNo())
}
