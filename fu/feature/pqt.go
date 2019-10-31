package feature

import (
	"github.com/google/logger"
	"github.com/sudachen/errors"
	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/reader"
	"github.com/xitongsys/parquet-go/source"
	"reflect"
	"runtime"
	"sync"
)

func ProcessPqtFile(fname string, f reflect.Value, t reflect.Type) (err error) {
	var (
		pf source.ParquetFile
		pr *reader.ParquetReader
	)

	NumCPU := runtime.NumCPU()

	if pf, err = local.NewLocalFileReader(fname); err != nil {
		return
	}
	defer pf.Close()
	if pr, err = reader.NewParquetReader(pf, reflect.New(t).Interface(), int64(runtime.NumCPU())); err != nil {
		return
	}
	defer pr.ReadStop()

	count := int(pr.GetNumRows())

	wg := sync.WaitGroup{}
	c := make(chan reflect.Value, NumCPU*2)
	haserr := false

	for i := 0; i < NumCPU; i++ {
		wg.Add(1)
		go func() {
			for v := range c {
				e := f.Call([]reflect.Value{v})
				if !e[0].IsNil() {
					logger.Errorf("error occured in Pqt processor: %s, file: %s",
						e[0].Interface().(error).Error(), fname)
					haserr = true
					break
				}
			}
			wg.Done()
		}()
	}

	wg2 := sync.WaitGroup{}
	MaxCountToRead := 1024
	L := MaxCountToRead
	for !haserr {
		if count < MaxCountToRead {
			if count == 0 {
				break
			}
			L = count
		}

		vs := reflect.MakeSlice(reflect.SliceOf(t), L, L)
		res := reflect.New(reflect.SliceOf(t))
		res.Elem().Set(vs)

		if err := pr.Read(res.Interface()); err != nil {
			haserr = true
			logger.Errorf("error occured in Pqt processor: %s, file: %s",
				err, fname)
			break
		}

		wg2.Wait()
		wg2.Add(1)
		count -= L
		go func(l reflect.Value, lLen int) {
			for j := 0; j < lLen && !haserr; j++ {
				c <- l.Index(j).Addr()
			}
			wg2.Done()
		}(res.Elem(), L)
	}

	wg2.Wait()
	close(c)
	wg.Wait()

	if haserr {
		return errors.Errorf("failed to process Pqt file %s", fname)
	}

	return
}

func ProcessPqt(fname string, f interface{}) (err error) {
	v := reflect.ValueOf(f)
	if v.Kind() != reflect.Func ||
		v.Type().NumIn() != 1 ||
		v.Type().NumOut() != 1 ||
		v.Type().In(0).Kind() != reflect.Ptr ||
		v.Type().In(0).Elem().Kind() != reflect.Struct ||
		v.Type().Out(0) != reflect.TypeOf((*error)(nil)).Elem() {

		return errors.Errorf("invalid pqt processor function type %v", v.Type())
	}

	return ProcessPqtFile(fname, v, v.Type().In(0).Elem())
}

