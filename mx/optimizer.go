package mx

import "github.com/sudachen/go-dnn/mx/capi"

func SgdMomUpdate(params, grads, mom *NDArray, lr, momentum, wd float32) error {
	return capi.OptimizerUpdate(
		capi.OpSgdUpdate,
		params.handle, grads.handle, mom.handle,
		capi.KeyLr, lr,
		capi.KeyMomentum, momentum,
		capi.KeyWd, wd)
}
