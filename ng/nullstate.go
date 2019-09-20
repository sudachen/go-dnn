package ng

import (
	"github.com/sudachen/go-dnn/nn"
)

type NullState struct {
	Metric    nn.Metric
	Epoch     int
	Optimizer nn.OptimizerConf
}

func (s *NullState) NextEpoch(maxEpochs int) (int, error) {
	if s.Epoch < maxEpochs || maxEpochs <= 0 {
		return s.Epoch, nil
	}
	return StopTraining, nil
}

func (s *NullState) Setup(net *nn.Network, seed int) (int, error) {
	if seed != 0 {
		seed = 42
	}
	net.Ctx.RandomSeed(seed)
	if err := net.Initialize(nil); err != nil {
		return 0, err
	}
	return seed, nil
}

func (s *NullState) Preset(net *nn.Network) (nn.Optimizer, error) {
	if s.Optimizer != nil {
		return s.Optimizer.Init(s.Epoch)
	}
	return nil, nil
}

func (s *NullState) LogBatchLoss(loss float32) error {
	return nil
}

func (s *NullState) FinishEpoch(net *nn.Network, test Batchs) (metric float32, satisfied bool, err error) {
	if s.Metric != nil {
		if satisfied, err = Measure(net, test, s.Metric, Silent); err != nil {
			return
		}
		metric = s.Metric.Value()
	}
	s.Epoch++
	return
}
