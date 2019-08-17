package tests

import (
	"github.com/sudachen/go-dnn/mx/internal"
	"gotest.tools/assert"
	"testing"
)

func assertPanic(t *testing.T, f func()) {
	defer func() {
		assert.Assert(t, recover() != nil, "The code did not panic")}()
	f()
}

func Test_Context(t *testing.T) {
	assertPanic(t, func(){ internal.OpEmpty.Value() })
	assertPanic(t, func(){ internal.OpNoOp.Value() })
	assertPanic(t, func(){ internal.KeyNoKey.Value() })
}

