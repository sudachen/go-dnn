
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

	acc, _, err := gym.Train(mx.CPU, mnistConv0, nil)
	assert.NilError(t, err)
	assert.Assert(t, acc >= gym.Accuracy)
}

```
```text
=== RUN   Test_mnistConv0
Network Identity: 5080c6ac83cd87f6aa66c98ff7ec79f4d5ab683f
Symbol           | Operation           | Output        |  Params #
------------------------------------------------------------------
_input           | null                | (64,1,28,28)  |         0
Convolution01    | Convolution((5,5)/) | (64,20,24,24) |       520
sym05            | Activation(tanh)    | (64,20,24,24) |         0
sym06            | Pooling(max)        | (64,20,12,12) |         0
Convolution02    | Convolution((5,5)/) | (64,50,8,8)   |     25050
sym07            | Activation(tanh)    | (64,50,8,8)   |         0
sym08            | Pooling(max)        | (64,50,4,4)   |         0
FullyConnected03 | FullyConnected      | (64,128)      |    102528
sym09            | Activation(tanh)    | (64,128)      |         0
FullyConnected04 | FullyConnected      | (64,10)       |      1290
sym10            | SoftmaxActivation() | (64,10)       |         0
sym11            | BlockGrad           | (64,10)       |         0
sym12            | make_loss           | (64,10)       |         0
sym13            | pick                | (64,1)        |         0
sym14            | log                 | (64,1)        |         0
sym15            | _mul_scalar         | (64,1)        |         0
sym16            | mean                | (1)           |         0
sym17            | make_loss           | (1)           |         0
------------------------------------------------------------------
Total params: 129388
Epoch 0, batch 552, avg loss: 0.1568979772442169
Epoch 0, accuracy: 0.9807692
Achieved reqired accuracy 0.98
--- PASS: Test_mnistConv0 (10.73s)
```
