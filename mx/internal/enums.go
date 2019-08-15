package internal

type MxnetKey int

const (
	KeyEmpty MxnetKey = iota
	KeyLow
	KeyHigh
	KeyNoKey
)

func (k MxnetKey) Value() string {
	switch k {
	case KeyLow:
		return "low"
	case KeyHigh:
		return "high"
	}
	panic("mxnet parameters key out of range")
}

type MxnetOp int

const (
	OpEmpty MxnetOp = iota
	OpRandomUniform
	OpNoOp
)

func (o MxnetOp) Value() string {
	switch o {
	case OpRandomUniform:
		return "_random_uniform"
	}
	panic("mxnet operation out of range")
}
