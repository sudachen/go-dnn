package tests

import (
	"github.com/sudachen/go-dnn/fu"
	"gotest.tools/assert"
	"testing"
)

func Test_fu_Contains(t *testing.T) {
	s1 := []string{"a","b","c"}
	s2 := []int{0,100,9999}
	assert.Assert(t,fu.Contains(s1, "b"))
	assert.Assert(t,fu.Contains(s2, 9999))
	assert.Assert(t,!fu.Contains(s1, "e"))
	assert.Assert(t,!fu.Contains(s2, -1))
}
