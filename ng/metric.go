package ng

import (
	"fmt"
	"github.com/sudachen/go-dnn/fu"
	"math"
)

type Classification struct {
	Accuracy float32

	count, ok int
}

func (c *Classification) Collect(data, label []float32) {
	if len(label) != 1 {
		panic(fmt.Sprintf("must have label as slice with one value"))
	}
	index := int(label[0])
	if len(data) < index || index < 0 {
		panic(fmt.Sprintf("label index out of data range"))
	}
	maxindex := index
	for i, v := range data {
		if v > data[maxindex] {
			maxindex = i
		}
	}
	c.count++
	if maxindex == index {
		c.ok++
	}
}

func (c *Classification) Reset() {
	c.count = 0
	c.ok = 0
}

func (c *Classification) Value() float32 {
	return float32(c.ok) / float32(c.count)
}

func (c *Classification) Satisfy() bool {
	return c.Value() >= c.Accuracy
}

type Erfc struct {
	Accuracy float32

	count int
	vals  float64
}

func (g *Erfc) Collect(data, label []float32) {
	var m float64
	for i, v := range label {
		//m += math.Erfc(math.Abs(float64(v - data[i]))/math.Abs(float64(v)))
		m += math.Erfc(math.Abs(float64(v - data[i])))
	}
	g.vals += m / float64(len(label))
	g.count++
}

func (g *Erfc) Reset() {
	g.vals = 0
	g.count = 0
}

func (g *Erfc) Value() float32 {
	return float32(g.vals / float64(g.count))
}

func (g *Erfc) Satisfy() bool {
	return g.Accuracy > 0 && g.Value() >= g.Accuracy
}

func ErfcAbs(scale, output, label float32) float64 {
	dif := math.Abs(float64(label) - float64(output))
	return math.Erfc(dif*float64(scale))
}

type DetailedMetric struct {
	Vals  []float64
	Count []int
	Accuracy float32
	Scale float32
	F     func(float32,float32,float32) float64
}

func (g *DetailedMetric) Collect(data, label []float32) {
	L := len(label)
	if g.Vals == nil || g.Count == nil {
		g.Vals = make([]float64,L+1)
		g.Count = make([]int,L+1)
	}
	f := g.F
	if f == nil {
		f = ErfcAbs
	}
	var m float64
	for i, v := range label {
		q := f(fu.IfZero(g.Scale,1),v,data[i])
		g.Vals[i] += q
		g.Count[i]++
		m += q
	}
	g.Vals[L] += m / float64(L)
	g.Count[L]++
}

func (g *DetailedMetric) Reset() {
	for i := range g.Vals {
		g.Vals[i] = 0
		g.Count[i] = 0
	}
}

func (g *DetailedMetric) Value() float32 {
	if g.Vals == nil {
		return 0
	}
	return float32(g.Vals[len(g.Vals)-1] / float64(g.Count[len(g.Vals)-1]))
}

func (g *DetailedMetric) Detail(i int) float32 {
	return float32(g.Vals[i] / float64(g.Count[i]))
}

func (g *DetailedMetric) Details() []float32 {
	if g.Vals != nil {
		r := make([]float32, len(g.Vals)-1)
		for i := range r {
			r[i] = float32(g.Vals[i] / float64(g.Count[i]))
		}
		return r
	}
	return []float32{}
}

func (g *DetailedMetric) Satisfy() bool {
	return g.Accuracy > 0 && g.Value() >= g.Accuracy
}

