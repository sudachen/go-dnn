package fu

import (
	"os"
	"path"
	"path/filepath"
)

const cacheGoDnn = ".cache/go-dnn"

var fullCacheDir string

func init() {
	if u, ok := os.LookupEnv("HOME"); ok {
		fullCacheDir, _ = filepath.Abs(filepath.Join(u, cacheGoDnn))
	} else {
		fullCacheDir, _ = filepath.Abs(cacheGoDnn)
	}
}

func CacheDir(d string) string {
	r := path.Join(fullCacheDir, d)
	_ = os.MkdirAll(r, 0777)
	return r
}

func CacheFile(f string) string {
	r := path.Join(fullCacheDir, f)
	_ = os.MkdirAll(path.Dir(r), 0777)
	return r
}
