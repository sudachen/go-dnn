package fu

import "math"

func Round10(a float64, digits int) float64 {
	q := math.Pow(10, float64(digits))
	return math.Round(a*q) / q
}

func Floor10(a float64, digits int) float64 {
	q := math.Pow(10, float64(digits))
	return math.Floor(a*q) / q
}

func Round(a []float32, digits int) []float32 {
	r := make([]float32, len(a))
	for i, v := range a {
		r[i] = float32(Round10(float64(v), digits))
	}
	return r
}

func Floor(a []float32, digits int) []float32 {
	r := make([]float32, len(a))
	for i, v := range a {
		r[i] = float32(Floor10(float64(v), digits))
	}
	return r
}
