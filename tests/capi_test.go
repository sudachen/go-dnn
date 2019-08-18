package tests

import (
	"github.com/sudachen/go-dnn/mx/capi"
	"gotest.tools/assert"
	"testing"
	"unsafe"
)

func Test_Capi_Cache(t *testing.T) {
	a := map[int]unsafe.Pointer{}
	for i := 0; i < capi.MaxCacheArgsCount; i++ {
		a[i] = unsafe.Pointer(capi.Cache(i))
	}
	for i := 0; i < capi.MaxCacheArgsCount; i++ {
		assert.Assert(t, a[i] == unsafe.Pointer(capi.Cache(i)))
	}
}
