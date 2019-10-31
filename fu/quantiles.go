package fu

import "math"

func Quantiles(count, L int, get func(i int) float32 ) (r []float32) {
	r = make([]float32,0,count)
	L = L-1
	x := float64(L)/float64(count)
	for i :=1; i < count; i++ {
		q := x*float64(i)
		left := get(int(math.Floor(q)))
		right := get(int(math.Ceil(q)))
		r = append(r, left+(right-left)*float32(q))
	}
	r = append(r,get(L))
	return
}
