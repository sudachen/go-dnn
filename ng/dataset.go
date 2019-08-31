package ng

type Dataset interface {
	Open(seed int, batchSize int) (Batchs, Batchs, error)
}

type Batchs interface {
	Next() bool
	Data() []float32
	Label() []float32
	Skip(int) error
	Close() error
	Reset() error
}
