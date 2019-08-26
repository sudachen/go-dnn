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
}

func (gym *Gym) verbose(s string) {
	if gym.Verbose == Printing {
		fmt.Println(s)
	} else if gym.Verbose == Logging {
		logger.Info(s)
	}
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

type store struct {
	GymStore
}

type epoch struct {
	GymEpoch
}

func (s store) EpochsCount() int {
	if s.GymStore != nil {
		return s.GymStore.EpochsCount()
	}
	return 0
}

func (s store) AddEpoch(i int) (epoch, error) {
	if s.GymStore != nil {
		ep, e := s.GymStore.AddEpoch(i)
		return epoch{ep}, e
	}
	return epoch{nil}, nil
}

func (s epoch) WriteBatchLoss(loss float64) error {
	if s.GymEpoch != nil {
		return s.GymEpoch.WriteBatchLoss(float32(loss))
	}
	return nil
}

func (s epoch) Commit() error {
	if s.GymEpoch != nil {
		return s.GymEpoch.Commit()
	}
	return nil
}

func (s epoch) Finish(accuracy float64, net *nn.Network) error {
	if s.GymEpoch != nil {
		err := s.GymEpoch.Finish(float32(accuracy), net)
		if err != nil {
			return err
		}
	}
	return nil
}
