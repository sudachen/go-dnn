package ng

import (
	"fmt"
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
	return g.Value() >= g.Accuracy
}
