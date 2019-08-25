package ng

import (
	"github.com/sudachen/go-dnn/mx"
	"io"
)

type WorkoutIdentity [32]byte

type S3 struct {
	Host       string
	Bucket     string
	Key        string
	Secret     string
	AutoUpload bool
}

type Workout struct {
	Id       string
	Path     string
	Seed     int64
	Continue bool

	Inherit string
	Input   mx.Dimension

	S3 *S3

	readonly bool
	identity *WorkoutIdentity
}

func (wrk *Workout) CopyParamsTo(w io.Writer) error {
	return nil
}

func (wrk *Workout) List() ([]*Workout, error) {
	return nil, nil
}

func (wrk *Workout) Close() error {
	return nil
}

func (wrk *Workout) Download() error {
	return nil
}

func (wrk *Workout) Upload() error {
	return nil
}
