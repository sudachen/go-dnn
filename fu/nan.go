package fu

import "math"

func IfNaN(a, dflt float32) float32 {
	q := float64(a)
	if math.IsNaN(q) || math.IsInf(q, -1) || math.IsInf(q, 0) {
		return dflt
	}
	return a
}
