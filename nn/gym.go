package nn

import (
	"fmt"
	"github.com/google/logger"
	"github.com/sudachen/go-dnn/mx"
	"time"
)

type GymWorkout interface {
	Get(idenity mx.GraphIdentity) (GymStore, error)
}

type GymStore interface {
	Seed() int
	EpochsCount() int
	AddEpoch(int) (GymEpoch, error)
	InitParams(params *Network) error
	WriteSummary(summary mx.Summary, seed int) error
}

type GymEpoch interface {
	WriteBatchLoss(loss float32) error
	Finish(accuracy float32, params *Network) error
	Commit() error
}

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

type GymDataset interface {
	Open(seed int, batchSize int) (GymBatchs, GymBatchs, error)
}

type GymBatchs interface {
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
	Dataset   GymDataset
	Verbose   GymVerbosity
	AccFunc   AccFunc
	Accuracy  float32
	Workout   GymWorkout
	Seed      int

	Network *Network
	store   GymStore
	epoch   GymEpoch
	seed    int
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

	gym.seed = gym.Seed

	if gym.seed == 0 {
		gym.seed = int(time.Now().Unix())
	}

	if gym.Workout != nil {

		if gym.store, err = gym.Workout.Get(gym.Network.Graph.Identity()); err != nil {
			return err
		}

		if gym.store.EpochsCount() == 0 {
			summary, err := gym.Network.Graph.Summary(true)
			if err != nil {
				return err
			}
			if err = gym.store.WriteSummary(summary, gym.seed); err != nil {
				return err
			}
		} else {
			gym.seed = gym.store.Seed()
		}

		gym.Network.Graph.Ctx.RandomSeed(gym.seed)
		if err = gym.store.InitParams(gym.Network); err != nil {
			return err
		}
	} else {
		gym.Network.Graph.Ctx.RandomSeed(gym.seed)
		if err = gym.Network.Graph.Initialize(nil); err != nil {
			return err
		}
	}

	return nil
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

func (gym *Gym) Train() (float32, error) {
	var (
		li, ti        GymBatchs
		err           error
		loss, acc     float64
		batchs, count int
	)

	if li, ti, err = gym.Dataset.Open(gym.seed+1, gym.BatchSize); err != nil {
		return 0, err
	}

	g := gym.Network.Graph
	lf := make([]float32, g.Loss.Dim().Total())

	for epoch := gym.startEpoch(); epoch < gym.Epochs; epoch++ {

		if err = gym.nextEpoch(epoch); err != nil {
			return 0, err
		}

		if err = li.Reset(); err != nil {
			return 0, err
		}

		sprint := gym.sprintOn()
		tm := sprint(0)

		gym.Network.Graph.Ctx.RandomSeed(gym.seed + epoch + 2)
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
			_ = gym.writeBatchLoss(loss)

			if tm != 0 {
				acc += loss
				count++
				if tx := sprint(tm); tx != tm {
					tm = tx
					loss = acc / float64(count)
					acc, count = 0, 0
					gym.verbose(fmt.Sprintf("Epoch %d, batch %d, avg loss: %v", epoch, batchs, loss))
					_ = gym.commitEpoch()
				}
			}
		}

		if err = ti.Reset(); err != nil {
			return 0, err
		}

		acc, count = 0, 0
		for ti.Next() {
			var a float64
			if a, err = gym.Network.Test(ti.Data(), ti.Label(), gym.AccFunc); err != nil {
				return 0, err
			}
			acc += a
			count++
		}

		acc = acc / float64(count)
		gym.verbose(fmt.Sprintf("Epoch %d, accuracy: %v", epoch, float32(acc)))

		if err := gym.finishEpoch(acc); err != nil {
			return 0, err
		}

		if gym.Accuracy > 0 && float32(acc) >= gym.Accuracy {
			gym.verbose(fmt.Sprintf("Achieved reqired accuracy %v", float32(gym.Accuracy)))
			break
		}
	}

	return float32(acc), nil
}
