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

var mnistMLP0 = nn.Connect(
	&nn.FullyConnected{Size: 128, Activation: nn.ReLU},
	&nn.FullyConnected{Size: 64, Activation: nn.ReLU},
	&nn.FullyConnected{Size: 10, Activation: nn.Softmax})

func Test_mnistMLP0(t *testing.T) {

	gym := &ng.Gym{
		Optimizer: &nn.Adam{Lr: .001, Loss: &nn.LabelCrossEntropyLoss{}},
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

	err := gym.Bind(mx.CPU, mnistMLP0)
	assert.NilError(t, err)
	t.Logf("Network Identity: %v", gym.Network.Identity())
	acc, err := gym.Train()
	assert.NilError(t, err)
	t.Logf("Accuracy: %v", acc)
	assert.Assert(t, acc >= gym.Accuracy)
}

