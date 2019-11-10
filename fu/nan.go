package fu

import "math"

func IfNaN(a, dflt float32) float32 {
	q := float64(a)
	if math.IsNaN(q) || math.IsInf(q, -1) || math.IsInf(q, 0) {
		return dflt
	}
	return a
}

func HasNaN(l []float32) bool {
	for _, v := range l {
		q := float64(v)
		if math.IsNaN(q) || math.IsInf(q, -1) || math.IsInf(q, 0) {
			return true
		}
	}
	return false
}

func CountZeros(l []float32) int {
	var count int
	for _, v := range l {
		if v == 0 { // not initialized
			count++
		}
	}
	return count
}
