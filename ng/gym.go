package ng

import (
	"fmt"
	"github.com/google/logger"
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"time"
)

type Verbosity int

const (
	_Silent Verbosity = iota
	Printing
	Logging
	NoSummary = 0x1000
	Silent = _Silent | NoSummary
)

type Gym struct {
	Optimizer nn.OptimizerConf
	Loss      mx.Loss
	Input     mx.Dimension
	Epochs    int
	Dataset   Dataset
	Verbose   Verbosity
	Every     time.Duration
	Metric    nn.Metric
	State     State
	Seed      int
}

const StopTraining = -1

type State interface {
	Setup(*nn.Network, int) (int, error)
	Preset(*nn.Network) (nn.Optimizer, error)
	LogBatchLoss(loss float32) error
	NextEpoch(maxEpochs int) (int, error)
	FinishEpoch(net *nn.Network, test Batchs) (float32, bool, error)
}

func verbose(s string, verbosity Verbosity) {
	if verbosity&0x3 == Printing {
		fmt.Println(s)
	} else if verbosity&0x3 == Logging {
		logger.Info(s)
	}
}

func (gym *Gym) verbose(s string) {
	verbose(s, gym.Verbose)
}

func (gym *Gym) everyTime() func(int64) int64 {

	interval := int64(gym.Every / time.Second)
	startedAt := time.Now().Unix()

	return func(tm int64) int64 {
		if interval == 0 {
			return tm
		}
		now := time.Now().Unix()
		if now > tm {
			tm = ((now-startedAt)/interval)*interval + interval + startedAt
		}
		return tm
	}
}
