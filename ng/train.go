package ng

import (
	"fmt"
)

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
