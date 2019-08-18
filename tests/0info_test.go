package tests

import (
	"github.com/sudachen/go-dnn/mx"
	"gotest.tools/assert"
	"testing"
)

var test_on_GPU = mx.GpuCount() > 0 && false
var RequiredMxVersion = mx.MakeVersion(1, 5, 0)

func Test_0info(t *testing.T) {
	t.Logf("go-dlnn version %v", mx.Version)
	t.Logf("libmxnet version %v", mx.LibVersion())
	t.Logf("GPUs count %v", mx.GpuCount())
	s := ""
	if !test_on_GPU {
		s = " not"
	}
	t.Logf("will%s test on GPU", s)
}

func Test_Version(t *testing.T) {
	assert.Assert(t, RequiredMxVersion == mx.Version)
}

func assertPanic(t *testing.T, f func()) {
	defer func() {
		assert.Assert(t, recover() != nil, "The code did not panic")
	}()
	f()
}
