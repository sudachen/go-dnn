package ng

import (
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"time"
)

func StateParams(gymState GymState) (nn.Params, error) {
	rd, err := gymState.OpenParams()
	if err != nil {
		return nn.Params{}, nil
	}
	defer rd.Close()
	p := nn.Params{}
	if err = p.Read(rd); err != nil {
		return nn.Params{}, nil
	}
	return p, nil
}

func (gym *Gym) init(net *nn.Network, workout GymWorkout) (state, int, error) {

	var (
		err      error
		count    int
		seed     int
		gymstate GymState
		summary  mx.Summary
		newOne   bool
	)

	seed = gym.Seed

	if seed == 0 {
		seed = int(time.Now().Unix())
	}

	if workout != nil {

		if summary, err = net.Summary(true); err != nil {
			return state{}, 0, err
		}

		if count, err = workout.Count(net.Identity()); err != nil {
			return state{}, 0, err
		}

		if gym.Continue && count > 0 {
			// get most recent
			if gymstate, err = workout.Get(net.Identity(), 0); err != nil {
				return state{}, 0, err
			}
			seed = gymstate.Seed()
		} else {
			// create new one
			if gymstate, err = workout.New(net.Identity(), seed, summary); err != nil {
				return state{}, 0, err
			}
			newOne = true
		}

		net.Ctx.RandomSeed(seed)
		if !newOne {
			p, err := StateParams(gymstate)
			if err != nil {
				return state{}, 0, err
			}
			if err = p.Setup(net, false); err != nil {
				return state{}, 0, err
			}
		}
	} else {
		net.Graph.Ctx.RandomSeed(seed)
		if err = net.Initialize(nil); err != nil {
			return state{}, 0, err
		}
	}

	return state{gymstate}, seed, nil
}
