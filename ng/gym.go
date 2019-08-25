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

	Network *nn.Network
	store   GymStore
	epoch   GymEpoch
	seed    int
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

func (gym *Gym) startEpoch() int {
	if gym.store != nil {
		return gym.store.EpochsCount()
	}
	return 0
}

func (gym *Gym) nextEpoch(i int) error {
	var err error
	if gym.store != nil {
		if gym.epoch, err = gym.store.AddEpoch(i); err != nil {
			return err
		}
	}
	return nil
}

func (gym *Gym) writeBatchLoss(loss float64) error {
	if gym.epoch != nil {
		return gym.epoch.WriteBatchLoss(float32(loss))
	}
	return nil
}

func (gym *Gym) commitEpoch() error {
	if gym.epoch != nil {
		return gym.epoch.Commit()
	}
	return nil
}

func (gym *Gym) finishEpoch(accuracy float64) error {
	if gym.epoch != nil {
		err := gym.epoch.Finish(float32(accuracy), gym.Network)
		gym.epoch = nil
		if err != nil {
			return err
		}
	}
	return nil
}

