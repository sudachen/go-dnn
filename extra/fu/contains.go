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
	} else if cv.Kind() == reflect.Map {
		return Contains(cv.MapRange(), val)
	}
	return false
}
