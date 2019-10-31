package fu

import "reflect"

func Contains(cont interface{}, val interface{}) bool {
	cv := reflect.ValueOf(cont)
	if cv.Kind() == reflect.Slice || cv.Kind() == reflect.Array {
		for i := 0; i < cv.Len(); i++ {
			if cv.Index(i).Interface() == val {
				return true
			}
		}
	}
	return false
}

func ValsOf(m interface{}) interface{} {
	v := reflect.ValueOf(m)
	if v.Kind() != reflect.Map {
		panic("parameter is not a map")
	}
	k := v.MapKeys()
	vals := reflect.MakeSlice(reflect.SliceOf(v.Type().Elem()), len(k), len(k))
	for i, s := range k {
		vals.Index(i).Set(v.MapIndex(s))
	}
	return vals.Interface()
}
