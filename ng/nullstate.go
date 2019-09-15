package ng

import (
	"fmt"
	"github.com/sudachen/go-dnn/nn"
)

type nullState struct {
	nn.AccFunc
	epochs int
}

func (s *nullState) NextEpoch(maxEpochs int) (int, error) {
	if s.epochs < maxEpochs {
		return s.epochs, nil
	}
	return StopTraining, nil
}

func (s *nullState) Setup(net *nn.Network, seed int) (int, error) {
	if seed != 0 {
		seed = 42
	}
	net.Ctx.RandomSeed(seed)
	if err := net.Initialize(nil); err != nil {
		return 0, err
	}
	return seed, nil
}

func (s *nullState) Preset(net *nn.Network) error {
	return nil
}

func (s *nullState) LogBatchLoss(loss float32) error {
	return nil
}

func (s *nullState) GetAccFunc() nn.AccFunc {
	return s.AccFunc
}

func (s *nullState) FinishEpoch(accuracy float32, net *nn.Network) error {
	s.epochs++
	return nil
}

func (s *nullState) LoadLastParams(*nn.Network) error {
	return fmt.Errorf("LoadLastParams is not nimplemented")
}
