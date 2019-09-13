package fu

import "math"

func Sum(a ...float32) float32 {
	var r float32 = 0
	for _, v := range a{
		r += v
	}
	return r
}

func Avg(a ...float32) float32 {
	var r float64 = 0
	for _, v := range a{
		r += float64(v)
	}
	return float32(r/float64(len(a)))
}

func Min(a ...float32) float32 {
	var r float32 = math.MaxFloat32
	for _, v := range a{
		if v < r {
			r = v
		}
	}
	return r
}

func Max(a ...float32) float32 {
	var r float32 = -math.MaxFloat32
	for _, v := range a{
		if v > r {
			r = v
		}
	}
	return r
}

func AbsMax(a ...float32) float32 {
	var r float32 = 0
	for _, v := range a{
		if v < 0 {
			v = -v
		}
		if v > r {
			r = v
		}
	}
	return r
}

func MinI(a ...int) int {
	var r int = math.MaxInt32
	for _, v := range a{
		if v < r {
			r = v
		}
	}
	return r
}

func SubMul(c float32, a,b []float32) []float32 {
	r := make([]float32,MinI(len(a),len(b)))
	for i := range r {
		r[i] = (a[i] - b[i])*c
	}
	return r
}

func AddMul(c float32, a,b []float32) []float32 {
	r := make([]float32,MinI(len(a),len(b)))
	for i := range r {
		r[i] = (a[i] + b[i])*c
	}
	return r
}

func FindMin(a []float32) (int, float32) {
	var r float32 = math.MaxFloat32
	var j int
	for i, v := range a{
		if v < r {
			j = i
			r = v
		}
	}
	return j, r
}

func FindMax(a []float32) (int, float32) {
	var r float32 = -math.MaxFloat32
	var j int
	for i, v := range a{
		if v > r {
			j = i
			r = v
		}
	}
	return j, r
}

func IfElse(t bool, a,b float32) float32 {
	if t {
		return a
	}
	return b
}
