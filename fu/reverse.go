package fu

import (
	"fmt"
	"reflect"
)

func Reverse(slice interface{}) {
	v := reflect.ValueOf(slice)
	if v.Kind() != reflect.Slice && v.Kind() != reflect.Array {
		panic(fmt.Sprintf("slice must have a slice/array type, but %v",v.Type()))
	}
	L := v.Len()
	for i := 0; i <= int((L-1)/2); i++ {
		reverseIndex := L - 1 - i
		tmp := v.Index(reverseIndex).Interface()
		v.Index(reverseIndex).Set(v.Index(i))
		v.Index(i).Set(reflect.ValueOf(tmp))
	}
}

func ReversedCopy(slice interface{}) interface{} {
	v := reflect.ValueOf(slice)
	if v.Kind() != reflect.Slice && v.Kind() != reflect.Array {
		panic(fmt.Sprintf("slice must have a slice/array type, but %v",v.Type()))
	}
	L := v.Len()
	o := reflect.MakeSlice(reflect.SliceOf(v.Type().Elem()), L, L)
	reflect.Copy(o,v)
	Reverse(o.Interface())
	return o.Interface()
}
