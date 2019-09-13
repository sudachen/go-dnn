package ng

type Dataset interface {
	Open(batchSize int) (Batchs, Batchs, error)
}

type Batchs interface {
	Next() bool
	Data() []float32
	Label() []float32
	Close() error
	Reset() error
	Randomize(seed int)
}
