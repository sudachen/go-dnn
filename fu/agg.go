package fu

import "math"

func Sum(a ...float32) float32 {
	var r float32 = 0
	for _, v := range a {
		r += v
	}
	return r
}

func Avg(a ...float32) float32 {
	var r float64 = 0
	for _, v := range a {
		r += float64(v)
	}
	return float32(r / float64(len(a)))
}

func Min(a ...float32) float32 {
	var r float32 = math.MaxFloat32
	for _, v := range a {
		if v < r {
			r = v
		}
	}
	return r
}

func Max(a ...float32) float32 {
	var r float32 = -math.MaxFloat32
	for _, v := range a {
		if v > r {
			r = v
		}
	}
	return r
}

func AbsMax(a ...float32) float32 {
	var r float32 = 0
	for _, v := range a {
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
	for _, v := range a {
		if v < r {
			r = v
		}
	}
	return r
}

func SubMul(c float32, a, b []float32) []float32 {
	r := make([]float32, MinI(len(a), len(b)))
	for i := range r {
		r[i] = (a[i] - b[i]) * c
	}
	return r
}

func AddMul(c float32, a, b []float32) []float32 {
	r := make([]float32, MinI(len(a), len(b)))
	for i := range r {
		r[i] = (a[i] + b[i]) * c
	}
	return r
}

func FindMin(a []float32) (int, float32) {
	var r float32 = math.MaxFloat32
	var j int
	for i, v := range a {
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
	for i, v := range a {
		if v > r {
			j = i
			r = v
		}
	}
	return j, r
}

func IfElse(t bool, a, b float32) float32 {
	if t {
		return a
	}
	return b
}

func IfElseI(t bool, a, b int) int {
	if t {
		return a
	}
	return b
}

func IfZero(a, b float32) float32 {
	if a > -1e-8 && a < 1e-8 {
		return b
	}
	return a
}

func Equal(a, b float32) bool {
	q := a-b
	return q > -1e-8 && q < 1e-8
}

func IfZeroI(a, b int) int {
	if a == 0 {
		return b
	}
	return a
}

/*
* MaxInRange retruns a if a in range (L,R] and R otherwise
 */
func MaxInRange(a, L, R float32) float32 {
	if a <= L || a > R {
		return R
	}
	return a
}

/*
* MaxInRangeI retruns a if a in range (L,R] and R otherwise
 */
func MaxInRangeI(a, L, R int) int {
	if a <= L || a > R {
		return R
	}
	return a
}

/*
* MinInRange retruns a if a in range [L,R) and L otherwise
 */
func MinInRange(a, L, R float32) float32 {
	if a < L || a >= R {
		return L
	}
	return a
}

/*
* MinInRangeL retruns a if a in range [L,R) and L otherwise
 */
func MinInRangeI(a, L, R int) int {
	if a < L || a >= R {
		return L
	}
	return a
}

func SoftMax(a []float32) (b []float32) {
	max := float64(Max(a...))
	var sum_exp_c float64
	for _, e := range a {
		sum_exp_c += math.Exp(float64(e) - max)
	}
	b = make([]float32, len(a))
	for i, v := range a {
		b[i] = float32(math.Exp(float64(v)-max) / sum_exp_c)
	}
	return
}

func Abs(a float32) float32 {
	return float32(math.Abs(float64(a)))
}
