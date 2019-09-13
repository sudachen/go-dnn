package fu

import (
	"reflect"
	"sort"
)

func SortedDictKeys(m interface{}) []string {
	v := reflect.ValueOf(m)
	if v.Kind() != reflect.Map || v.Type().Key() != reflect.TypeOf("") {
		panic("parameter is not a map")
	}
	keys := KeysOf(m).([]string)
	sort.Strings(keys)
	return keys
}

func KeysOf(m interface{}) interface{} {
	v := reflect.ValueOf(m)
	if v.Kind() != reflect.Map {
		panic("parameter is not a map")
	}
	k := v.MapKeys()
	keys := reflect.MakeSlice(reflect.SliceOf(v.Type().Key()),len(k),len(k))
	for i, s := range k {
		keys.Index(i).Set(s)
	}
	return keys.Interface()
}

func MakeSetFrom(keys interface{}) interface{} {
	cv := reflect.ValueOf(keys)
	if cv.Kind() == reflect.Slice || cv.Kind() == reflect.Array {
		m := reflect.MakeMap(reflect.MapOf(reflect.TypeOf(keys).Elem(),reflect.TypeOf(true)))
		t := reflect.ValueOf(true)
		for i := 0; i < cv.Len(); i++ {
			m.SetMapIndex(cv.Index(i),t)
		}
		return m.Interface()
	}
	panic("parameter is not a map")
}
