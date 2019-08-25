
[![CircleCI](https://circleci.com/gh/sudachen/go-dnn.svg?style=svg)](https://circleci.com/gh/sudachen/go-dnn)
[![Maintainability](https://api.codeclimate.com/v1/badges/5af58cfe8b9efbe0f29f/maintainability)](https://codeclimate.com/github/sudachen/go-dnn/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/5af58cfe8b9efbe0f29f/test_coverage)](https://codeclimate.com/github/sudachen/go-dnn/test_coverage)
[![Go Report Card](https://goreportcard.com/badge/github.com/sudachen/go-dnn)](https://goreportcard.com/report/github.com/sudachen/go-dnn)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


```golang
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
		Optimizer: &nn.Adam{Lr: .001, Loss: &nn.LabelCrossEntropyLoss{}},
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

	err := gym.Bind(mx.CPU, mnistConv0)
	assert.NilError(t, err)
	t.Logf("Network Identity: %v", gym.Network.Graph.Identity())
	acc, err := gym.Train()
	assert.NilError(t, err)
	t.Logf("Accuracy: %v", acc)
	assert.Assert(t, acc >= gym.Accuracy)
}

```
