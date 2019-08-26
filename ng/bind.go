package ng

import (
	"github.com/sudachen/go-dnn/nn"
	"time"
)

func (gym *Gym) init(net *nn.Network, workout GymWorkout) (store, int, error) {

	var (
		err  error
		s    store
		seed int
	)

	seed = gym.Seed

	if seed == 0 {
		seed = int(time.Now().Unix())
	}

	if workout != nil {

		gymstor, err := workout.Get(net.Identity())
		if err != nil {
			return s, 0, err
		}
		s = store{gymstor}

		if s.EpochsCount() == 0 {
			summary, err := net.Summary(true)
			if err != nil {
				return s, 0, err
			}
			if err = s.WriteSummary(summary, seed); err != nil {
				return s, 0, err
			}
		} else {
			seed = s.Seed()
		}

		net.Ctx.RandomSeed(seed)
		if err = s.InitParams(net); err != nil {
			return s, 0, err
		}
	} else {
		net.Graph.Ctx.RandomSeed(seed)
		if err = net.Initialize(nil); err != nil {
			return s, 0, err
		}
	}

	return s, seed, nil
}
