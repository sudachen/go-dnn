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

var mnistMLP0 = nn.Connect(
	&nn.FullyConnected{Size: 128, Activation: nn.ReLU},
	&nn.FullyConnected{Size: 64, Activation: nn.ReLU},
	&nn.FullyConnected{Size: 10, Activation: nn.Softmax})

func Test_mnistMLP0(t *testing.T) {

	gym := &ng.Gym{
		Optimizer: &nn.Adam{Lr: .001},
		Loss:      &nn.LabelCrossEntropyLoss{},
		BatchSize: 32,
		Input:     mx.Dim(1, 28, 28),
		Epochs:    5,
		Verbose:   ng.Printing,
		Every:     1 * time.Second,
		Dataset:   &mnist.Dataset{},
		AccFunc:   ng.Classification,
		Accuracy:  0.96,
		Seed:      42,
	}

	acc, params, err := gym.Train(mx.CPU, mnistMLP0)
	assert.NilError(t, err)
	assert.Assert(t, acc >= 0.96)

	net1, err := nn.Bind(mx.CPU, mnistMLP0, mx.Dim(50, 1, 28, 28), nil)
	assert.NilError(t, err)
	defer net1.Release()
	_ = net1.PrintSummary(false)
	err = net1.SetParams(params, false)
	assert.NilError(t, err)
	acc, err = ng.Measure(net1, &mnist.Dataset{}, ng.Classification, ng.Printing)
	assert.Assert(t, acc >= 0.96)
	err = net1.SaveParamsFile(fu.CacheFile("tests/mnistMLP0.params"))
	assert.NilError(t, err)

	net2, err := nn.Bind(mx.CPU, mnistMLP0, mx.Dim(10, 1, 28, 28), nil)
	assert.NilError(t, err)
	err = net2.LoadParamsFile(fu.CacheFile("tests/mnistMLP0.params"), false)
	assert.NilError(t, err)
	acc, err = ng.Measure(net2, &mnist.Dataset{}, ng.Classification, ng.Printing)
	assert.Assert(t, acc >= 0.96)
}
