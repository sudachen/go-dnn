package tests

import (
	"github.com/sudachen/go-dnn/mx"
	"testing"
)

func Test_0info(t *testing.T) {
	t.Logf("go-dlnn version %v", mx.Version)
	t.Logf("libmxnet version %v", mx.LibVersion())
	t.Logf("GPUs count %v", mx.GpuCount())
}

