package ng

import "fmt"

type AvgLoss struct {
	Avg   float64
	Count int
	Hist  []float32
	index int
}

func (a *AvgLoss) Reset() {
	for i := range a.Hist {
		a.Hist[i] = 0
	}
	a.Avg = 0
	a.Count = 0
}

func (a *AvgLoss) Value() float32 {
	return float32(a.Avg)
}

func (a *AvgLoss) Last() float32 {
	var (
		acc   float32
		count int
	)
	if len(a.Hist) > 0 {
		for _, v := range a.Hist {
			if v != 0 {
				acc += v
				count++
			}
		}
		return acc / float32(count)
	}
	return 0
}

func (a *AvgLoss) Add(val float32) {
	if len(a.Hist) > 0 {
		a.Hist[a.index] = val
		a.index = (a.index + 1) % len(a.Hist)
	}
	a.Avg = (a.Avg*float64(a.Count) + float64(val)) / float64(a.Count+1)
	a.Count += 1
}

func (a AvgLoss) String() string {
	return fmt.Sprintf("loss: %.4f(%d)/%.4f", a.Last(), len(a.Hist), a.Value())
}
