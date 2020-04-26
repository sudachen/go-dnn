
[![Go Report Card](https://goreportcard.com/badge/github.com/sudachen/go-dnn)](https://goreportcard.com/report/github.com/sudachen/go-dnn)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


It's an old version of dnn fo golang. Please see new updated version https://github.com/go-ml-dev/nn

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
```
```text
Network Identity: 158cf5bd604e12e7bd438084e135703bd89dc10f
Symbol              | Operation            | Output        |  Params #
----------------------------------------------------------------------
_input              | null                 | (32,1,28,28)  |         0
Convolution01       | Convolution((3,3)//) | (32,24,26,26) |       240
Convolution01$A     | Activation(relu)     | (32,24,26,26) |         0
MaxPool02           | Pooling(max)         | (32,24,13,13) |         0
Convolution03       | Convolution((5,5)//) | (32,32,9,9)   |     19232
Convolution03$BN    | BatchNorm            | (32,32,9,9)   |       128
Convolution03$A     | Activation(relu)     | (32,32,9,9)   |         0
MaxPool04           | Pooling(max)         | (32,32,4,4)   |         0
FullyConnected05    | FullyConnected       | (32,32)       |     16416
FullyConnected05$BN | BatchNorm            | (32,32)       |       128
sigmoid@sym07       | sigmoid              | (32,32)       |         0
FullyConnected05$A  | elemwise_mul         | (32,32)       |         0
FullyConnected06    | FullyConnected       | (32,10)       |       330
FullyConnected06$A  | SoftmaxActivation()  | (32,10)       |         0
BlockGrad@sym08     | BlockGrad            | (32,10)       |         0
make_loss@sym09     | make_loss            | (32,10)       |         0
pick@sym10          | pick                 | (32,1)        |         0
log@sym11           | log                  | (32,1)        |         0
_mul_scalar@sym12   | _mul_scalar          | (32,1)        |         0
mean@sym13          | mean                 | (1)           |         0
make_loss@sym14     | make_loss            | (1)           |         0
----------------------------------------------------------------------
Total params: 36474
[000] batch: 389, loss: 0.09991227
[000] batch: 1074, loss: 0.055281825
[000] batch: 1855, loss: 0.0760978
[000] metric: 0.988, final loss: 0.0515
Achieved reqired metric
Symbol              | Operation            | Output        |  Params #
----------------------------------------------------------------------
_input              | null                 | (10,1,28,28)  |         0
Convolution01       | Convolution((3,3)//) | (10,24,26,26) |       240
Convolution01$A     | Activation(relu)     | (10,24,26,26) |         0
MaxPool02           | Pooling(max)         | (10,24,13,13) |         0
Convolution03       | Convolution((5,5)//) | (10,32,9,9)   |     19232
Convolution03$BN    | BatchNorm            | (10,32,9,9)   |       128
Convolution03$A     | Activation(relu)     | (10,32,9,9)   |         0
MaxPool04           | Pooling(max)         | (10,32,4,4)   |         0
FullyConnected05    | FullyConnected       | (10,32)       |     16416
FullyConnected05$BN | BatchNorm            | (10,32)       |       128
sigmoid@sym07       | sigmoid              | (10,32)       |         0
FullyConnected05$A  | elemwise_mul         | (10,32)       |         0
FullyConnected06    | FullyConnected       | (10,10)       |       330
FullyConnected06$A  | SoftmaxActivation()  | (10,10)       |         0
----------------------------------------------------------------------
Total params: 36474
Accuracy over 1000*10 batchs: 0.988
--- PASS: Test_mnistConv0 (6.51s)
```
