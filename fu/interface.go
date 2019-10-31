package fu

import "reflect"

func Iarray(a interface{}) []interface{} {
	v := reflect.ValueOf(a)
	r := make([]interface{}, v.Len())
	for i := range r {
		r[i] = v.Index(i).Interface()
	}
	return r
}
