package nn

import (
	"fmt"
	"github.com/google/logger"
	"github.com/sudachen/go-dnn/mx"
	"time"
)

func ClassifyAccuracy(data, label []float32) (bool, error) {
	if len(label) != 1 {
		return false, fmt.Errorf("Classify must have label parameter as slice with one value")
	}
	index := int(label[0])
	if len(data) < index || index < 0 {
		return false, fmt.Errorf("Classify label index out of data range")
	}
	maxindex := index
	for i, v := range data {
		if v > data[maxindex] {
			maxindex = i
		}
	}
	return maxindex == index, nil
}

type Dataset interface {
	Open(seed int64, batchSize int) (BatchsIterator, BatchsIterator, error)
}

type BatchsIterator interface {
	Next() bool
	Data() []float32
	Label() []float32
	Notes() []string
	Skip(int) error
	Close() error
	Reset() error
}

type GymVerbosity int

const (
	GymSilent GymVerbosity = iota
	GymPrint
	GymLog
)

type Gym struct {
	Optimizer OptimizerConf
	BatchSize int
	Input     mx.Dimension
	Epochs    int
	Sprint    time.Duration
	Dataset   Dataset
	Verbose   GymVerbosity
	AccFunc   AccFunc
	Accuracy  float32

	Network *Network
	Workout *Workout
}

func (gym *Gym) verbose(s string) {
	if gym.Verbose == GymPrint {
		fmt.Println(s)
	} else if gym.Verbose == GymLog {
		logger.Info(s)
	}
}

func (gym *Gym) Bind(ctx mx.Context, nb Block) error {
	var err error

	input := gym.Input.Push(gym.BatchSize)

	if gym.Network, err = Bind(ctx, nb, input, gym.Optimizer); err != nil {
		return err
	}

	if gym.Verbose != GymSilent {
		_ = gym.Network.Graph.SummaryOut(true, gym.verbose)
	}

	return nil
}

func sprintOn(dur time.Duration) func(int64) int64 {

	sprint := int64(dur / time.Second)
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

func (gym *Gym) Train() (float32, error) {
	var (
		li, ti        BatchsIterator
		err           error
		seed          int64
		sprint        = sprintOn(gym.Sprint)
		tm            = sprint(0)
		loss, acc     float64
		batchs, count int
	)

	if li, ti, err = gym.Dataset.Open(seed, gym.BatchSize); err != nil {
		return 0, err
	}

	g := gym.Network.Graph
	lf := make([]float32, g.Loss.Dim().Total())

	for epoch := 0; epoch < gym.Epochs; epoch++ {
		if err = li.Reset(); err != nil {
			return 0, err
		}

		acc, count, batchs = 0, 0, 0
		for li.Next() {
			batchs++
			if err = gym.Network.Train(li.Data(), li.Label()); err != nil {
				return 0, err
			}

			if err = g.Loss.CopyValuesTo(lf); err != nil {
				return 0, err
			}
			loss = 0
			for _, v := range lf {
				loss += float64(v)
			}
			loss /= float64(len(lf))

			if tm != 0 {
				acc += loss
				count++
				if tx := sprint(tm); tx != tm {
					tm = tx
					loss = acc / float64(count)
					acc, count = 0, 0
					gym.verbose(fmt.Sprintf("Epoch %d, batch %d, avg loss: %v", epoch, batchs, loss))
				}
			}
		}

		if err = ti.Reset(); err != nil {
			return 0, err
		}

		acc, count = 0, 0
		for ti.Next() {
			var a float64
			if a, err = gym.Network.Test(li.Data(), li.Label(), gym.AccFunc); err != nil {
				return 0, err
			}
			acc += a
			count++
		}

		acc = acc / float64(count)
		gym.verbose(fmt.Sprintf("Epoch %d, accuracy: %v", epoch, float32(acc)))
		if gym.Accuracy > 0 && float32(acc) >= gym.Accuracy {
			gym.verbose(fmt.Sprintf("Achieved reqired accuracy %v", float32(gym.Accuracy)))
			break
		}
	}

	return float32(acc), nil
}
