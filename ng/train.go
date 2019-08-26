package ng

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"time"
)

func (gym *Gym) Train(ctx mx.Context, nb nn.Block, workout GymWorkout) (float32, nn.Params, error) {
	var (
		li, ti        GymBatchs
		err           error
		loss, acc     float32
		batchs, count int
		net           *nn.Network
		st            state
		ep            epoch
		seed          int
		opt           nn.Optimizer
		params        nn.Params
	)

	defer func() {
		if opt != nil {
			opt.Release()
		}
		if net != nil {
			net.Release()
		}
	}()

	input := gym.Input.Push(gym.BatchSize)

	if net, err = nn.Bind(ctx, nb, input, gym.Loss); err != nil {
		return 0, nn.Params{}, err
	}

	gym.verbose(fmt.Sprintf("Network Identity: %v", net.Identity()))

	if gym.Verbose != Silent {
		_ = net.SummaryOut(true, gym.verbose)
	}

	if st, seed, err = gym.init(net, workout); err != nil {
		return 0, nn.Params{}, err
	}

	if li, ti, err = gym.Dataset.Open(seed+1, gym.BatchSize); err != nil {
		return 0, nn.Params{}, err
	}
	defer func() { _ = li.Close(); _ = ti.Close() }()

	lf := make([]float32, net.Loss.Dim().Total())

	for epoch := st.EpochsCount(); epoch < gym.Epochs; epoch++ {

		if opt, err = gym.Optimizer.Init(); err != nil {
			return 0, nn.Params{}, err
		}

		if ep, err = st.AddEpoch(epoch); err != nil {
			return 0, nn.Params{}, err
		}

		if err = li.Reset(); err != nil {
			return 0, nn.Params{}, err
		}

		sprint := gym.sprintOn()
		tm := sprint(0)

		net.Ctx.RandomSeed(seed + epoch + 2)
		acc, count, batchs = 0, 0, 0
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
			_ = ep.WriteBatchLoss(loss)

			if tm != 0 {
				acc += loss
				count++
				if tx := sprint(tm); tx != tm {
					tm = tx
					loss = acc / float32(count)
					acc, count = 0, 0
					gym.verbose(fmt.Sprintf("Epoch %d, batch %d, avg loss: %v", epoch, batchs, loss))
					_ = ep.Commit()
				}
			}
		}

		opt.Release()
		opt = nil

		if acc, err = Measure(net, gym.Dataset, gym.AccFunc, Silent); err != nil {
			return 0, nn.Params{}, err
		}

		gym.verbose(fmt.Sprintf("Epoch %d, accuracy: %v", epoch, float32(acc)))

		if workout != nil {
			if params, err = nn.ParamsOf(net); err != nil {
				return 0, nn.Params{}, err
			}
		}

		if err := ep.Finish(acc, params); err != nil {
			return 0, nn.Params{}, err
		}

		if gym.Accuracy > 0 && float32(acc) >= gym.Accuracy {
			gym.verbose(fmt.Sprintf("Achieved reqired accuracy %v", float32(gym.Accuracy)))
			break
		}
	}

	if workout == nil {
		if params, err = nn.ParamsOf(net); err != nil {
			return 0, nn.Params{}, err
		}
	}

	return float32(acc), params, nil
}

func Measure(net *nn.Network, batches interface{}, accfunc nn.AccFunc, verbosity Verbosity) (float32, error) {

	var (
		err    error
		li, ti GymBatchs
		acc    float32
		count  int
	)

	switch s := batches.(type) {
	case GymBatchs:
		ti = s
	case GymDataset:
		if li, ti, err = s.Open(int(time.Now().Unix()), net.Input.Len(0)); err != nil {
			return 0, err
		}
		defer func() { _ = li.Close(); _ = ti.Close() }()
	default:
		return 0, fmt.Errorf("samples for Measure function must be ng.GymBatchs or ng.Dataset")
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
		acc += a
		count++
	}

	acc = acc / float32(count)

	verbose(fmt.Sprintf("Accuracy over %d batches: %v", count, float32(acc)), verbosity)
	return acc, nil
}
