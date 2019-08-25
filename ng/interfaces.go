package ng

import (
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
)

type GymWorkout interface {
	Get(idenity mx.GraphIdentity) (GymStore, error)
}

type GymStore interface {
	Seed() int
	EpochsCount() int
	AddEpoch(int) (GymEpoch, error)
	InitParams(params *nn.Network) error
	WriteSummary(summary mx.Summary, seed int) error
}

type GymEpoch interface {
	WriteBatchLoss(loss float32) error
	Finish(accuracy float32, params *nn.Network) error
	Commit() error
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
