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
	Silent Verbosity = iota
	Printing
	Logging
)

type Gym struct {
	Optimizer nn.OptimizerConf
	Loss      mx.Loss
	Input     mx.Dimension
	Epochs    int
	Dataset   Dataset
	Verbose   Verbosity
	Every     time.Duration
	AccFunc   nn.AccFunc
	Accuracy  float32
	State     State
	Seed      int
}

const StopTraining = -1

type State interface {
	Setup(*nn.Network,int) (int,error)
	Preset(*nn.Network) error
	LogBatchLoss(loss float32) error
	NextEpoch(maxEpochs int) (int, error)
	FinishEpoch(accuracy float32, net *nn.Network) error
}

func verbose(s string, verbosity Verbosity) {
	if verbosity == Printing {
		fmt.Println(s)
	} else if verbosity == Logging {
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

