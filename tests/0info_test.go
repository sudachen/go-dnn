package tests

import (
	"github.com/sudachen/go-dnn/mx"
	"testing"
)

var test_on_GPU = mx.GpuCount() > 0 && false

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
