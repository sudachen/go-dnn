package ng

import (
	"fmt"
	"github.com/sudachen/go-dnn/fu"
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"time"
)

const avgLossLen = 10

type avgLoss struct {
	index int
	hist  [avgLossLen]float32
}

func (a *avgLoss) Value() float32 {
	var (
		acc   float32
		count int
	)
	for _, v := range a.hist {
		if v != 0 {
			acc += v
			count++
		}
	}
	return acc / float32(count)
}

func (a *avgLoss) Add(val float32) {
	a.hist[a.index] = val
	a.index = (a.index + 1) % len(a.hist)
}

func (gym *Gym) Train(ctx mx.Context, nb nn.Block) (float32, nn.Params, error) {
	var (
		li, ti    Batchs
		err       error
		loss, acc float32
		epoch     int
		batchs    int
		net       *nn.Network
		st        nullState
		seed      int
		opt       nn.Optimizer
		params    nn.Params
		state     State
	)

	defer func() {
		if opt != nil {
			opt.Release()
		}
		if net != nil {
			net.Release()
		}
		if li != nil {
			_ = li.Close()
		}
		if ti != nil {
			_ = ti.Close()
		}
	}()

	nextTime := gym.everyTime()
	input := gym.Input.Push(gym.BatchSize)

	if net, err = nn.Bind(ctx, nb, input, gym.Loss); err != nil {
		return 0, nn.Params{}, err
	}

	gym.verbose(fmt.Sprintf("Network Identity: %v", net.Identity()))

	if gym.Verbose != Silent {
		_ = net.SummaryOut(true, gym.verbose)
	}

	if gym.State != nil  {
		state = gym.State
	} else {
		state = &nullState{}
	}

	if gym.Seed == 0 {
		seed = int(time.Now().Unix())
	} else {
		seed = gym.Seed
	}

	if seed, err = state.Setup(net,seed); err != nil {
		return 0, nn.Params{}, err
	}

	if li, ti, err = gym.Dataset.Open(seed+1, gym.BatchSize); err != nil {
		return 0, nn.Params{}, err
	}

	lf := make([]float32, net.Loss.Dim().Total())

	for {

		if epoch, err = st.NextEpoch(gym.Epochs); err != nil {
			return 0, nn.Params{}, err
		}

		if epoch <= StopTraining || (gym.Epochs > 0 && epoch >= gym.Epochs) {
			break
		}

		avg := avgLoss{}

		if opt, err = gym.Optimizer.Init(); err != nil {
			return 0, nn.Params{}, err
		}

		if err = li.Reset(); err != nil {
			return 0, nn.Params{}, err
		}

		if err = state.Preset(net); err != nil {
			return 0, nn.Params{}, err
		}

		tm := nextTime(0)

		net.Ctx.RandomSeed(seed + epoch + 2)
		acc, batchs = 0, 0
		for li.Next() {
			batchs++
			if err = net.Train(li.Data(), li.Label(), opt); err != nil {
				return 0, nn.Params{}, err
			}

			if err = net.Loss.CopyValuesTo(lf); err != nil {
				return 0, nn.Params{}, err
			}
			loss = 0
			for _, v := range lf {
				loss += v
			}
			loss /= float32(len(lf))
			_ = state.LogBatchLoss(loss)

			avg.Add(loss)

			if tm != 0 {
				if tx := nextTime(tm); tx != tm {
					tm = tx
					gym.verbose(fmt.Sprintf("Epoch %d, batch %d, loss: %v", epoch, batchs, avg.Value()))
				}
			}
		}

		opt.Release()
		opt = nil

		if err := state.FinishEpoch(acc, net); err != nil {
			return 0, nn.Params{}, err
		}

		if acc, err = Measure(net, ti, gym.AccFunc, Silent); err != nil {
			return 0, nn.Params{}, err
		}

		gym.verbose(fmt.Sprintf("Epoch %d, accuracy: %v, final loss: %v", epoch, fu.Round1(acc,3), fu.Round1(avg.Value(),4)))

		if gym.Accuracy > 0 && acc >= gym.Accuracy {
			gym.verbose(fmt.Sprintf("Achieved reqired accuracy %v", gym.Accuracy))
			break
		}
	}

	if params, err = net.GetParams(); err != nil {
		return 0, nn.Params{}, err
	}

	return float32(acc), params, nil
}

func Measure(net *nn.Network, batchs interface{}, accfunc nn.AccFunc, verbosity Verbosity) (float32, error) {

	var (
		err    error
		li, ti Batchs
		acc    float64
		count  int
	)

	switch s := batchs.(type) {
	case Batchs:
		ti = s
	case Dataset:
		if li, ti, err = s.Open(int(time.Now().Unix()), net.Input.Len(0)); err != nil {
			return 0, err
		}
		defer func() { _ = li.Close(); _ = ti.Close() }()
	default:
		return 0, fmt.Errorf("samples for Measure function must be ng.Batchs or ng.Dataset")
	}

	if err = ti.Reset(); err != nil {
		return 0, err
	}

	acc, count = 0, 0
	for ti.Next() {
		var a float32
		if a, err = net.Test(ti.Data(), ti.Label(), accfunc); err != nil {
			return 0, err
		}
		acc += float64(a)
		count++
	}

	w := net.Input.Len(0)
	acc = acc / float64(count)

	verbose(fmt.Sprintf("Accuracy over %d*%d batchs: %v", count, w, fu.Round1(float32(acc),3)), verbosity)
	return float32(acc), nil
}
