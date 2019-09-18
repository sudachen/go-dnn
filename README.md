
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
Network Identity: 19f5235a3e21c185702fc580808dcac2b00cc955
Symbol              | Operation            | Output        |  Params #
----------------------------------------------------------------------
_input              | null                 | (32,1,28,28)  |         0
Convolution01       | Convolution((3,3)//) | (32,24,26,26) |       240
Convolution01$A     | Activation(relu)     | (32,24,26,26) |         0
Pooling@sym05       | Pooling(max)         | (32,24,13,13) |         0
Convolution02       | Convolution((5,5)//) | (32,32,9,9)   |     19232
Convolution02$BN    | BatchNorm            | (32,32,9,9)   |       128
Convolution02$A     | Activation(relu)     | (32,32,9,9)   |         0
Pooling@sym06       | Pooling(max)         | (32,32,4,4)   |         0
FullyConnected03    | FullyConnected       | (32,32)       |     16416
FullyConnected03$BN | BatchNorm            | (32,32)       |       128
sigmoid@sym07       | sigmoid              | (32,32)       |         0
FullyConnected03$A  | elemwise_mul         | (32,32)       |         0
FullyConnected04    | FullyConnected       | (32,10)       |       330
FullyConnected04$A  | SoftmaxActivation()  | (32,10)       |         0
BlockGrad@sym08     | BlockGrad            | (32,10)       |         0
make_loss@sym09     | make_loss            | (32,10)       |         0
pick@sym10          | pick                 | (32,1)        |         0
log@sym11           | log                  | (32,1)        |         0
_mul_scalar@sym12   | _mul_scalar          | (32,1)        |         0
mean@sym13          | mean                 | (1)           |         0
make_loss@sym14     | make_loss            | (1)           |         0
----------------------------------------------------------------------
Total params: 36474
Epoch 0, batch 369, loss: 0.12668017
Epoch 0, batch 1183, loss: 0.05189542
Epoch 0, accuracy: 0.988, final loss: 0.0515
Achieved reqired accuracy 0.98
Symbol              | Operation            | Output        |  Params #
----------------------------------------------------------------------
_input              | null                 | (10,1,28,28)  |         0
Convolution01       | Convolution((3,3)//) | (10,24,26,26) |       240
Convolution01$A     | Activation(relu)     | (10,24,26,26) |         0
Pooling@sym05       | Pooling(max)         | (10,24,13,13) |         0
Convolution02       | Convolution((5,5)//) | (10,32,9,9)   |     19232
Convolution02$BN    | BatchNorm            | (10,32,9,9)   |       128
Convolution02$A     | Activation(relu)     | (10,32,9,9)   |         0
Pooling@sym06       | Pooling(max)         | (10,32,4,4)   |         0
FullyConnected03    | FullyConnected       | (10,32)       |     16416
FullyConnected03$BN | BatchNorm            | (10,32)       |       128
sigmoid@sym07       | sigmoid              | (10,32)       |         0
FullyConnected03$A  | elemwise_mul         | (10,32)       |         0
FullyConnected04    | FullyConnected       | (10,10)       |       330
FullyConnected04$A  | SoftmaxActivation()  | (10,10)       |         0
----------------------------------------------------------------------
Total params: 36474
Accuracy over 1000*10 batchs: 0.988
--- PASS: Test_mnistConv0 (5.90s)
```
