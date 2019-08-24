package fu

import (
	"reflect"
	"sort"
)

func SortedDictKeys(m interface{}) []string {
	v := reflect.ValueOf(m)
	if v.Kind() != reflect.Map {
		panic("parameter is not a map")
	}
	k := v.MapKeys()
	keys := make([]string, len(k))
	for i, s := range k {
		keys[i] = s.String()
	}
	sort.Strings(keys)
	return keys
}
