package ng

import (
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"io"
)

type GymWorkout interface {
	Name() string
	Identities() ([]mx.GraphIdentity, error)
	// zero index points to most recent state
	Count(idenity mx.GraphIdentity) (int, error)
	Get(idenity mx.GraphIdentity, index int) (GymState, error)
	New(idenity mx.GraphIdentity, seed int, summary mx.Summary) (GymState, error)
}

type GymState interface {
	Seed() int
	EpochsCount() int
	AddEpoch(int) (GymEpoch, error)
	ReadSummary() (mx.Summary, error)
	OpenParams() (io.ReadCloser, error)
}

type GymEpoch interface {
	OpenBatchLoss() (io.ReadCloser, error)
	WriteBatchLoss(loss float32) error
	Finish(accuracy float32, params nn.Params) error
	Commit() error
}

type GymDataset interface {
	Open(seed int, batchSize int) (GymBatchs, GymBatchs, error)
}

type GymBatchs interface {
	Next() bool
	Data() []float32
	Label() []float32
	Skip(int) error
	Close() error
	Reset() error
}
