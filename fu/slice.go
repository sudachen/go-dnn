package fu

func Slice(a []float32, width, column int) []float32 {
	rows := len(a)/width
	r := make([]float32,rows)
	for i := range r {
		r[i] = a[width*i+column]
	}
	return r
}
