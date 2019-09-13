package fu

import (
	"fmt"
	"os"
)

func Fatalf(s string, a ...interface{}) {
	_, _ = fmt.Fprintf(os.Stderr, s, a...)
	os.Exit(1)
}


