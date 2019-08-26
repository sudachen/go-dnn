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
		BatchSize: 64,
		Input:     mx.Dim(1, 28, 28),
		Epochs:    5,
		Sprint:    5 * time.Second,
		Verbose:   ng.Printing,
		Dataset:   &mnist.Dataset{},
		AccFunc:   ng.Classification,
		Accuracy:  0.96,
		Seed:      42,
	}

	acc, params, err := gym.Train(mx.CPU, mnistMLP0, nil)
	assert.NilError(t, err)
	assert.Assert(t, acc >= gym.Accuracy)

	net1, err := nn.Bind(mx.CPU, mnistMLP0, mx.Dim(16, 1, 28, 28), nil)
	assert.NilError(t, err)
	defer net1.Release()
	_ = net1.PrintSummary(false)
	err = params.Setup(net1, false)
	assert.NilError(t, err)
	acc, err = ng.Measure(net1, &mnist.Dataset{}, ng.Classification, ng.Printing)
	assert.Assert(t, acc >= gym.Accuracy)
	err = params.Save(fu.CacheFile("tests/mnistMLP0.params"))
	assert.NilError(t, err)

	params2, err := nn.LoadParams(fu.CacheFile("tests/mnistMLP0.params"))
	assert.NilError(t, err)
	net2, err := nn.Bind(mx.CPU, mnistMLP0, mx.Dim(16, 1, 28, 28), nil)
	err = params2.Setup(net2, false)
	assert.NilError(t, err)
	acc, err = ng.Measure(net2, &mnist.Dataset{}, ng.Classification, ng.Printing)
	assert.Assert(t, acc >= gym.Accuracy)
}
