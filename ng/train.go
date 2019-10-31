package ng

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"time"
)

func (gym *Gym) Train(ctx mx.Context, nb nn.Block) (metric float32, params nn.Params, err error) {
	var (
		li, ti    Batchs
		loss      float32
		epoch     int
		batchs    int
		net       *nn.Network
		seed      int
		opt       nn.Optimizer
		state     State
		satisfied bool
		avg       = AvgLoss{Hist: make([]float32, 10)}
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

	if net, err = nn.Bind(ctx, nb, gym.Input, gym.Loss); err != nil {
		return
	}

	gym.verbose(fmt.Sprintf("Network Identity: %v", net.Identity()))
	if gym.Verbose&NoSummary == 0 {
		_ = net.SummaryOut(true, gym.verbose)
	}

	if gym.State != nil {
		state = gym.State
	} else {
		state = &NullState{Metric: gym.Metric, Optimizer: gym.Optimizer}
	}

	if gym.Seed == 0 {
		seed = int(time.Now().Unix())
	} else {
		seed = gym.Seed
	}

	if seed, err = state.Setup(net, seed); err != nil {
		return
	}

	if li, ti, err = gym.Dataset.Open(net.BatchSize); err != nil {
		return
	}

	lf := make([]float32, net.Loss.Dim().Total())

	for {

		if epoch, err = state.NextEpoch(gym.Epochs); err != nil {
			return
		}

		if epoch <= StopTraining || (gym.Epochs > 0 && epoch >= gym.Epochs) {
			break
		}

		if gym.Verbose != Silent {
			avg.Reset()
		}

		li.Randomize(seed + epoch)
		if err = li.Reset(); err != nil {
			return
		}

		if opt, err = state.Preset(net); err != nil {
			return
		}

		tm := nextTime(0)

		net.Ctx.RandomSeed(seed + epoch + 2)
		batchs = 0
		for li.Next() {
			batchs++
			if err = net.Train(li.Data(), li.Label(), opt); err != nil {
				return
			}

			if err = net.Loss.CopyValuesTo(lf); err != nil {
				return
			}
			loss = 0
			for _, v := range lf {
				loss += v
			}
			loss /= float32(len(lf))
			_ = state.LogBatchLoss(loss)

			if gym.Verbose != Silent {
				avg.Add(loss)
			}

			if tm != 0 {
				if tx := nextTime(tm); tx != tm {
					tm = tx
					gym.verbose(fmt.Sprintf("[%03d] batch: %d, loss: %.4f", epoch, batchs, avg.Last()))
				}
			}
		}

		opt.Release()
		opt = nil

		ti.Randomize(seed + epoch)

		if metric, satisfied, err = state.FinishEpoch(net, ti); err != nil {
			return
		}

		gym.verbose(fmt.Sprintf("[%03d] metric: %.3f, avg/last loss: %.4f/%.4f", epoch, metric, avg.Avg, avg.Last()))

		if satisfied {
			gym.verbose("Achieved reqired metric")
			break
		}
	}

	if params, err = net.GetParams(); err != nil {
		return
	}

	return
}
