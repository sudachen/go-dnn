package fu

import "math"

func Round1(a float32, digits int) float32 {
	q := math.Pow(10, float64(digits))
	return float32(math.Round(float64(a)*q) / q)
}

func Floor1(a float32, digits int) float32 {
	q := math.Pow(10, float64(digits))
	return float32(math.Floor(float64(a)*q) / q)
}

func Round(a []float32, digits int) []float32 {
	r := make([]float32, len(a))
	for i, v := range a {
		r[i] = float32(Round1(v, digits))
	}
	return r
}

func Floor(a []float32, digits int) []float32 {
	r := make([]float32, len(a))
	for i, v := range a {
		r[i] = float32(Floor1(v, digits))
	}
	return r
}
