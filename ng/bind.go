package ng

import (
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"time"
)

func (gym *Gym) Bind(ctx mx.Context, nb nn.Block) error {
	var err error

	input := gym.Input.Push(gym.BatchSize)

	if gym.Network, err = nn.Bind(ctx, nb, input, gym.Optimizer); err != nil {
		return err
	}

	if gym.Verbose != Silent {
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

