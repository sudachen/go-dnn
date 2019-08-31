
[![CircleCI](https://circleci.com/gh/sudachen/go-dnn.svg?style=svg)](https://circleci.com/gh/sudachen/go-dnn)
[![Maintainability](https://api.codeclimate.com/v1/badges/5af58cfe8b9efbe0f29f/maintainability)](https://codeclimate.com/github/sudachen/go-dnn/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/5af58cfe8b9efbe0f29f/test_coverage)](https://codeclimate.com/github/sudachen/go-dnn/test_coverage)
[![Go Report Card](https://goreportcard.com/badge/github.com/sudachen/go-dnn)](https://goreportcard.com/report/github.com/sudachen/go-dnn)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


```golang
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
	&nn.Convolution{Channels: 20, Kernel: mx.Dim(5, 5), Activation: nn.Tanh},
	&nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	&nn.Convolution{Channels: 50, Kernel: mx.Dim(5, 5), Activation: nn.Tanh},
	&nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	&nn.FullyConnected{Size: 128, Activation: nn.Tanh},
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
		AccFunc:   ng.Classification,
		Accuracy:  0.98,
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

	acc, err = ng.Measure(net, &mnist.Dataset{}, ng.Classification, ng.Printing)
	assert.Assert(t, acc >= 0.98)
}
```
```text
=== RUN   Test_mnistConv0
Network Identity: 62b2117bed4ef63412495b6dfe582a065759cb17
Symbol           | Operation           | Output        |  Params #
------------------------------------------------------------------
_input           | null                | (32,1,28,28)  |         0
Convolution01    | Convolution((5,5)/) | (32,20,24,24) |       520
sym05            | Activation(tanh)    | (32,20,24,24) |         0
sym06            | Pooling(max)        | (32,20,12,12) |         0
Convolution02    | Convolution((5,5)/) | (32,50,8,8)   |     25050
sym07            | Activation(tanh)    | (32,50,8,8)   |         0
sym08            | Pooling(max)        | (32,50,4,4)   |         0
FullyConnected03 | FullyConnected      | (32,128)      |    102528
sym09            | Activation(tanh)    | (32,128)      |         0
FullyConnected04 | FullyConnected      | (32,10)       |      1290
sym10            | SoftmaxActivation() | (32,10)       |         0
sym11            | BlockGrad           | (32,10)       |         0
sym12            | make_loss           | (32,10)       |         0
sym13            | pick                | (32,1)        |         0
sym14            | log                 | (32,1)        |         0
sym15            | _mul_scalar         | (32,1)        |         0
sym16            | mean                | (1)           |         0
sym17            | make_loss           | (1)           |         0
------------------------------------------------------------------
Total params: 129388
Epoch 0, batch 226, loss: 0.084210135
Epoch 0, batch 597, loss: 0.08045147
Epoch 0, batch 976, loss: 0.080270246
Epoch 0, batch 1360, loss: 0.13755605
Epoch 0, batch 1747, loss: 0.06917396
Epoch 0, accuracy: 0.985, final loss: 0.0316
Achieved reqired accuracy 0.98
Symbol           | Operation           | Output        |  Params #
------------------------------------------------------------------
_input           | null                | (10,1,28,28)  |         0
Convolution01    | Convolution((5,5)/) | (10,20,24,24) |       520
sym05            | Activation(tanh)    | (10,20,24,24) |         0
sym06            | Pooling(max)        | (10,20,12,12) |         0
Convolution02    | Convolution((5,5)/) | (10,50,8,8)   |     25050
sym07            | Activation(tanh)    | (10,50,8,8)   |         0
sym08            | Pooling(max)        | (10,50,4,4)   |         0
FullyConnected03 | FullyConnected      | (10,128)      |    102528
sym09            | Activation(tanh)    | (10,128)      |         0
FullyConnected04 | FullyConnected      | (10,10)       |      1290
sym10            | SoftmaxActivation() | (10,10)       |         0
------------------------------------------------------------------
Total params: 129388
Accuracy over 1000*10 batchs: 0.985
--- PASS: Test_mnistConv0 (12.23s)
```
