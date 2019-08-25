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

func GymPredict0(gym *ng.Gym, skip int) ([]float32, []float32, error) {
	var (
		ti  ng.GymBatchs
		err error
		r   [][]float32
	)

	if _, ti, err = gym.Dataset.Open(0, gym.BatchSize); err != nil {
		return nil, nil, err
	}

	if err = ti.Skip(skip); err != nil {
		return nil, nil, err
	}

	ti.Next()
	if r, err = gym.Network.Predict(ti.Data()); err != nil {
		return nil, nil, err
	}

	l := ti.Label()
	return fu.Floor(r[0], 2), l[:len(l)/gym.BatchSize], nil
}

func f_gym(accuracy float32) *ng.Gym {
	return &ng.Gym{
		Optimizer: &nn.Adam{Lr: .001, Loss: &nn.LabelCrossEntropyLoss{}},
		BatchSize: 64,
		Input:     mx.Dim(1, 28, 28),
		Epochs:    5,
		Sprint:    5 * time.Second,
		Verbose:   ng.Printing,
		Dataset:   &mnist.Dataset{},
		AccFunc:   ng.ClassifyAccuracy,
		Accuracy:  accuracy,
		Seed:      42,
	}
}

func f_gym_train(t *testing.T, accuracy float32, ctx mx.Context, nb nn.Block) {
	gym := f_gym(accuracy)
	err := gym.Bind(ctx, nb)
	assert.NilError(t, err)
	t.Logf("Network Identity: %v", gym.Network.Graph.Identity())
	acc, err := gym.Train()
	assert.NilError(t, err)
	t.Logf("Accuracy: %v", acc)
	assert.Assert(t, acc >= gym.Accuracy)
	r, l, err := GymPredict0(gym, 0)
	assert.NilError(t, err)
	t.Log(r, l)
}

var mnist_Conv = nn.Connect(
	&nn.Convolution{Channels: 20, Kernel: mx.Dim(5, 5), Activation: nn.Tanh},
	&nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	&nn.Convolution{Channels: 50, Kernel: mx.Dim(5, 5), Activation: nn.Tanh},
	&nn.MaxPool{Kernel: mx.Dim(2, 2), Stride: mx.Dim(2, 2)},
	//&nn.Flatten{},
	&nn.FullyConnected{Size: 128, Activation: nn.Tanh},
	&nn.FullyConnected{Size: 10, Activation: nn.Softmax})

func Test_mnistConv_0(t *testing.T) {
	f_gym_train(t, 0.98, mx.CPU, mnist_Conv)
}

var mnist_MLP = nn.Connect(
	//&nn.Flatten{},
	&nn.FullyConnected{Size: 128, Activation: nn.ReLU},
	&nn.FullyConnected{Size: 64, Activation: nn.ReLU},
	&nn.FullyConnected{Size: 10, Activation: nn.Softmax})

func Test_mnistMLP(t *testing.T) {
	f_gym_train(t, 0.96, mx.CPU, mnist_MLP)
}
