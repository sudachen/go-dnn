package fu

import (
	"reflect"
	"unsafe"
)

func Index(i int, p interface{}) unsafe.Pointer {
	pv := reflect.ValueOf(p)
	of := pv.Elem().Type().Size() * uintptr(i)
	return unsafe.Pointer(pv.Pointer() + of)
}
