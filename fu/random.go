package fu

import "math/rand"

func RandomIndex(ln, seed int) []int {
	r := make([]int, ln)
	for i := range r {
		r[i] = i
	}
	rand.New(rand.NewSource(int64(seed))).Shuffle(
		ln,
		func(i, j int) { r[i], r[j] = r[j], r[i] })
	return r
}
