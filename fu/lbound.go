package fu

func LowerBoundary(ln int, less func(i int) bool) int {
	it := 0
	for ln > 0 {
		step := ln/2
		i := it + step
		if less(i) {
			it = i+1
			ln -= step+1
		} else {
			ln = step
		}
	}
	return it
}

