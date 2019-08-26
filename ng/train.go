package ng

import (
	"fmt"
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
)

func (gym *Gym) Train(ctx mx.Context, nb nn.Block, workout ...GymWorkout) (float32, error) {
	var (
		li, ti        GymBatchs
		err           error
		loss, acc     float64
		batchs, count int
		net           *nn.Network
		st            store
		ep            epoch
		seed          int
		wo            GymWorkout
		opt           nn.Optimizer
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
		return 0, err
	}

	gym.verbose(fmt.Sprintf("Network Identity: %v", net.Identity()))

	if gym.Verbose != Silent {
		_ = net.SummaryOut(true, gym.verbose)
	}

	if len(workout) > 0 {
		wo = workout[0]
	}

	if st, seed, err = gym.init(net, wo); err != nil {
		return 0, err
	}

	if li, ti, err = gym.Dataset.Open(seed+1, gym.BatchSize); err != nil {
		return 0, err
	}
	defer func() { _ = li.Close(); _ = ti.Close() }()

	lf := make([]float32, net.Loss.Dim().Total())

	for epoch := st.EpochsCount(); epoch < gym.Epochs; epoch++ {

		if opt, err = gym.Optimizer.Init(); err != nil {
			return 0, err
		}

		if ep, err = st.AddEpoch(epoch); err != nil {
			return 0, err
		}

		if err = li.Reset(); err != nil {
			return 0, err
		}

		sprint := gym.sprintOn()
		tm := sprint(0)

		net.Ctx.RandomSeed(seed + epoch + 2)
		acc, count, batchs = 0, 0, 0
		for li.Next() {
			batchs++
			if err = net.Train(li.Data(), li.Label(), opt); err != nil {
				return 0, err
			}

			if err = net.Loss.CopyValuesTo(lf); err != nil {
				return 0, err
			}
			loss = 0
			for _, v := range lf {
				loss += float64(v)
			}
			loss /= float64(len(lf))
			_ = ep.WriteBatchLoss(loss)

			if tm != 0 {
				acc += loss
				count++
				if tx := sprint(tm); tx != tm {
					tm = tx
					loss = acc / float64(count)
					acc, count = 0, 0
					gym.verbose(fmt.Sprintf("Epoch %d, batch %d, avg loss: %v", epoch, batchs, loss))
					_ = ep.Commit()
				}
			}
		}

		opt.Release()
		opt = nil

		if err = ti.Reset(); err != nil {
			return 0, err
		}

		acc, count = 0, 0
		for ti.Next() {
			var a float64
			if a, err = net.Test(ti.Data(), ti.Label(), gym.AccFunc); err != nil {
				return 0, err
			}
			acc += a
			count++
		}

		acc = acc / float64(count)
		gym.verbose(fmt.Sprintf("Epoch %d, accuracy: %v", epoch, float32(acc)))

		if err := ep.Finish(acc, net); err != nil {
			return 0, err
		}

		if gym.Accuracy > 0 && float32(acc) >= gym.Accuracy {
			gym.verbose(fmt.Sprintf("Achieved reqired accuracy %v", float32(gym.Accuracy)))
			break
		}
	}

	return float32(acc), nil
}
