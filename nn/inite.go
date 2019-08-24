package nn

import "github.com/sudachen/go-dnn/mx"

type Const struct {
	Value float32
}

func (x *Const) Inite(a *mx.NDArray) error {
	if x.Value == 0 {
		return a.Zeros().Err()
	}
	return a.Fill(x.Value).Err()
}

type XavierFactor int

const (
	XavierIn  XavierFactor = 1
	XavierOut XavierFactor = 2
	XavierAvg XavierFactor = 3
)

type Xavier struct {
	Gaussian  bool
	Magnitude float32
	Factor    XavierFactor
}

func (x *Xavier) Inite(a *mx.NDArray) error {
	var magnitude float32 = 3.
	if x.Magnitude > 0 {
		magnitude = x.Magnitude
	}
	factor := 2 // Avg
	if x.Factor >= XavierIn && x.Factor <= XavierAvg {
		factor = int(x.Factor)
	}
	return a.Xavier(x.Gaussian, factor, magnitude).Err()
}

type Uniform struct {
	Magnitude float32
}

func (x *Uniform) Inite(a *mx.NDArray) error {
	var magnitude float32 = 1.
	if x.Magnitude > 0 {
		magnitude = x.Magnitude
	}
	return a.Uniform(0, magnitude).Err()
}
