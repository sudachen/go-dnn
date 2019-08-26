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
	BatchSize int
	Input     mx.Dimension
	Epochs    int
	Sprint    time.Duration
	Dataset   GymDataset
	Verbose   Verbosity
	AccFunc   nn.AccFunc
	Accuracy  float32
	Workout   GymWorkout
	Seed      int
	Continue  bool
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

func (gym *Gym) sprintOn() func(int64) int64 {

	sprint := int64(gym.Sprint / time.Second)
	startedAt := time.Now().Unix()

	return func(tm int64) int64 {
		if sprint == 0 {
			return tm
		}
		now := time.Now().Unix()
		if now > tm {
			tm = ((now-startedAt)/sprint)*sprint + sprint + startedAt
		}
		return tm
	}
}

type state struct {
	GymState
}

type epoch struct {
	GymEpoch
}

func (s state) EpochsCount() int {
	if s.GymState != nil {
		return s.GymState.EpochsCount()
	}
	return 0
}

func (s state) AddEpoch(i int) (epoch, error) {
	if s.GymState != nil {
		ep, e := s.GymState.AddEpoch(i)
		return epoch{ep}, e
	}
	return epoch{nil}, nil
}

func (s epoch) WriteBatchLoss(loss float32) error {
	if s.GymEpoch != nil {
		return s.GymEpoch.WriteBatchLoss(loss)
	}
	return nil
}

func (s epoch) Commit() error {
	if s.GymEpoch != nil {
		return s.GymEpoch.Commit()
	}
	return nil
}

func (s epoch) Finish(accuracy float32, params nn.Params) error {
	if s.GymEpoch != nil {
		err := s.GymEpoch.Finish(accuracy, params)
		if err != nil {
			return err
		}
	}
	return nil
}
