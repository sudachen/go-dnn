package tests

import (
	"github.com/sudachen/go-dnn/data/mnist"
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/ng"
	"github.com/sudachen/go-dnn/nn"
	"gotest.tools/assert"
	"testing"
	"time"
)

var mnistConv0 = nn.Connect(
	&nn.Convolution{Channels: 20, Kernel: mx.Dim(5, 5), Activation: nn.Tanh},
	&nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	&nn.Convolution{Channels: 50, Kernel: mx.Dim(5, 5), Activation: nn.Tanh},
	&nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	&nn.FullyConnected{Size: 128, Activation: nn.Tanh},
	&nn.FullyConnected{Size: 10, Activation: nn.Softmax})

func Test_mnistConv0(t *testing.T) {

	gym := &ng.Gym{
		Optimizer: &nn.Adam{Lr: .001},
		Loss: 	   &nn.LabelCrossEntropyLoss{},
		BatchSize: 64,
		Input:     mx.Dim(1, 28, 28),
		Epochs:    5,
		Sprint:    5 * time.Second,
		Verbose:   ng.Printing,
		Dataset:   &mnist.Dataset{},
		AccFunc:   ng.Classification,
		Accuracy:  0.98,
		Seed:      42,
	}

	acc, err := gym.Train(mx.CPU, mnistConv0)
	assert.NilError(t, err)
	assert.Assert(t, acc >= gym.Accuracy)
}
