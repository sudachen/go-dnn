package tests

import (
	"fmt"
	"github.com/sudachen/go-dnn/fu"
	"gotest.tools/assert"
	"testing"
)

func Test_fu_Contains(t *testing.T) {
	s1 := []string{"a", "b", "c"}
	s2 := []int{0, 100, 9999}
	assert.Assert(t, fu.Contains(s1, "b"))
	assert.Assert(t, fu.Contains(s2, 9999))
	assert.Assert(t, !fu.Contains(s1, "e"))
	assert.Assert(t, !fu.Contains(s2, -1))
}

func fuCompare(a, b interface{}) bool {
	s0 := fmt.Sprintf("%v", a)
	s1 := fmt.Sprintf("%v", b)
	return s0 == s1
}

func Test_fu_Round(t *testing.T) {
	assert.Assert(t, fuCompare(fu.Round1(1.333222, 3), 1.333))
	assert.Assert(t, fuCompare(fu.Round([]float32{1.333222}, 3), []float32{1.333}))
	assert.Assert(t, fuCompare(fu.Round1(1.333666, 3), 1.334))
	assert.Assert(t, fuCompare(fu.Round([]float32{1.333666}, 3), []float32{1.334}))
}

func Test_fu_Floor(t *testing.T) {
	assert.Assert(t, fuCompare(fu.Floor1(1.333222, 3), 1.333))
	assert.Assert(t, fuCompare(fu.Floor([]float32{1.333222}, 3), []float32{1.333}))
	assert.Assert(t, fuCompare(fu.Floor1(1.333666, 3), 1.333))
	assert.Assert(t, fuCompare(fu.Floor([]float32{1.333666}, 3), []float32{1.333}))
}
