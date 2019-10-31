package fu

import "math/rand"

func RandomIndex(ln, seed int) []int {
	/*r := make([]int, ln)
	for i := range r {
		r[i] = i
	}
	rand.New(rand.NewSource(int64(seed))).Shuffle(
		ln,
		func(i, j int) { r[i], r[j] = r[j], r[i] })
	return r*/
	return rand.New(rand.NewSource(int64(seed))).Perm(ln)
}

func Uniform(ln, seed int, low, high float32) (a []float32) {
	a = make([]float32, ln)
	r := rand.New(rand.NewSource(int64(seed)))
	for i := range a {
		a[i] = r.Float32()*(low-high) + low
	}
	return
}
