package tests

import (
	"github.com/sudachen/go-dnn/data/mnist"
	"github.com/sudachen/go-dnn/fu"
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/ng"
	"github.com/sudachen/go-dnn/nn"
	"gotest.tools/assert"
	"testing"
	"time"
)

var mnistConv0 = nn.Connect(
	&nn.Convolution{Channels: 24, Kernel: mx.Dim(3, 3), Activation: nn.ReLU},
	&nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	&nn.Convolution{Channels: 32, Kernel: mx.Dim(5, 5), Activation: nn.ReLU, BatchNorm: true},
	&nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	&nn.FullyConnected{Size: 32, Activation: nn.Swish, BatchNorm: true},
	&nn.FullyConnected{Size: 10, Activation: nn.Softmax})

func Test_mnistConv0(t *testing.T) {

	gym := &ng.Gym{
		Optimizer: &nn.Adam{Lr: .001},
		Loss:      &nn.LabelCrossEntropyLoss{},
		Input:     mx.Dim(32, 1, 28, 28),
		Epochs:    5,
		Verbose:   ng.Printing,
		Every:     1 * time.Second,
		Dataset:   &mnist.Dataset{},
		Metric:    &ng.Classification{Accuracy: 0.98},
		Seed:      42,
	}

	acc, params, err := gym.Train(mx.CPU, mnistConv0)
	assert.NilError(t, err)
	assert.Assert(t, acc >= 0.98)
	err = params.Save(fu.CacheFile("tests/mnistConv0.params"))
	assert.NilError(t, err)

	net, err := nn.Bind(mx.CPU, mnistConv0, mx.Dim(10, 1, 28, 28), nil)
	assert.NilError(t, err)
	err = net.LoadParamsFile(fu.CacheFile("tests/mnistConv0.params"), false)
	assert.NilError(t, err)
	_ = net.PrintSummary(false)

	ok, err := ng.Measure(net, &mnist.Dataset{}, &ng.Classification{Accuracy: 0.98}, ng.Printing)
	assert.Assert(t, ok)
}
