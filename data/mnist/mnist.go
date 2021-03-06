package mnist

import (
	"bytes"
	"compress/gzip"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"github.com/sudachen/go-dnn/fu"
	"github.com/sudachen/go-dnn/ng"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const baseURL = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/mnist/"
const cacheDir = "datasets/mnist"

type dsFile struct {
	Name string
	Hash string
}

func (f dsFile) Verify(dir string) error {
	fullname := filepath.Join(dir, f.Name)
	if _, err := os.Stat(fullname); err != nil {
		return err
	}

	file, err := os.Open(fullname)
	if err != nil {
		return err
	}
	defer file.Close()

	SHA1 := sha1.New()
	if _, err := io.Copy(SHA1, file); err != nil {
		return err
	}

	h := hex.EncodeToString(SHA1.Sum(nil)[:20])
	if h != f.Hash {
		return fmt.Errorf("%v: SHA1 %v is not equal to %v", f.Name, h, f.Hash)
	}

	return nil
}

func (f dsFile) Download(dir string) error {
	if _, err := os.Stat(dir); err != nil {
		if err = os.MkdirAll(dir, 0777); err != nil {
			return err
		}
	}

	if err := f.Verify(dir); err == nil {
		return nil
	}

	resp, err := http.Get(baseURL + f.Name)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(filepath.Join(dir, f.Name))
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err = io.Copy(out, resp.Body); err != nil {
		return err
	}

	return f.Verify(dir)
}

func (f *dsFile) Load(dir string) ([]byte, error) {
	fd, err := os.Open(filepath.Join(dir, f.Name))
	if err != nil {
		return nil, err
	}

	gr, err := gzip.NewReader(fd)
	if err != nil {
		return nil, err
	}

	var bf bytes.Buffer
	if _, err := io.Copy(&bf, gr); err != nil {
		return nil, err
	}

	return bf.Bytes(), nil
}

var trainData = dsFile{"train-images-idx3-ubyte.gz", "6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d"}
var trainLabel = dsFile{"train-labels-idx1-ubyte.gz", "2a80914081dc54586dbdf242f9805a6b8d2a15fc"}
var testData = dsFile{"t10k-images-idx3-ubyte.gz", "c3a25af1f52dad7f726cce8cacb138654b760d48"}
var testLabel = dsFile{"t10k-labels-idx1-ubyte.gz", "763e7fa3757d93b0cdec073cef058b2004252c17"}

type Dataset struct{}

func (d Dataset) Open(batchSize int) (ng.Batchs, ng.Batchs, error) {
	for _, v := range []*dsFile{&trainData, &trainLabel, &testData, &testLabel} {
		if err := v.Download(fu.CacheDir(cacheDir)); err != nil {
			return nil, nil, err
		}
	}

	trainIter := &Batchs{BatchSize: batchSize}
	if err := trainIter.Load(&trainData, &trainLabel); err != nil {
		return nil, nil, err
	}

	testIter := &Batchs{BatchSize: batchSize}
	if err := testIter.Load(&testData, &testLabel); err != nil {
		return nil, nil, err
	}

	return trainIter, testIter, nil
}

type Batchs struct {
	BatchSize             int
	DataBs, LabelBs       []byte
	Count, Batchs         int
	DataLength            int
	LabelLength           int
	Index                 int
	RndIndex              []int
	DataBatch, LabelBatch []float32
}

func (b *Batchs) Randomize(seed int) {
	b.RndIndex = fu.RandomIndex(b.Batchs, seed)
}

func (b *Batchs) Load(dataFile, labelFile *dsFile) error {
	var (
		data, label []byte
		err         error
	)

	if data, err = dataFile.Load(fu.CacheDir(cacheDir)); err != nil {
		return err
	}
	if 0x00000803 != binary.BigEndian.Uint32(data) {
		return fmt.Errorf("not mnist data file")
	}
	if label, err = labelFile.Load(fu.CacheDir(cacheDir)); err != nil {
		return err
	}
	if 0x00000801 != binary.BigEndian.Uint32(label) {
		return fmt.Errorf("not mnist label file")
	}
	count := binary.BigEndian.Uint32(label[4:8])
	if count != binary.BigEndian.Uint32(data[4:8]) {
		return fmt.Errorf("incorrect samples count")
	}
	b.LabelBs = label[8:]
	b.DataBs = data[16:]
	b.Count = int(count)
	b.Batchs = b.Count / b.BatchSize
	b.DataLength = int(binary.BigEndian.Uint32(data[8:12]) * binary.BigEndian.Uint32(data[12:16]))
	b.LabelLength = 1

	b.LabelBatch = make([]float32, b.LabelLength*b.BatchSize)
	b.DataBatch = make([]float32, b.DataLength*b.BatchSize)
	return nil
}

func (b *Batchs) Next() bool {
	idx := b.Index
	if idx < b.Batchs {
		if b.RndIndex != nil {
			idx = b.RndIndex[idx]
		}
		width := b.DataLength * b.BatchSize
		src := b.DataBs[idx*width : (idx+1)*width]
		for i := 0; i < width; i++ {
			b.DataBatch[i] = float32(src[i]) / 255
		}
		width = b.LabelLength * b.BatchSize
		src = b.LabelBs[idx*width : (idx+1)*width]
		for i := 0; i < width; i++ {
			b.LabelBatch[i] = float32(src[i])
		}
		b.Index++
		return true
	}
	return false
}

func (b *Batchs) Data() []float32 {
	return b.DataBatch
}

func (b *Batchs) Label() []float32 {
	return b.LabelBatch
}

func (b *Batchs) Close() error {
	return nil
}

func (b *Batchs) Reset() error {
	b.Index = 0
	return nil
}
