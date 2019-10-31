package fu

import "fmt"

func Assert(expr bool, a ...interface{}) {
	if !expr {
		if len(a) > 0 {
			panic(fmt.Sprintf(a[0].(string), a[1:]...))
		}
		panic("assert failed")
	}
}
