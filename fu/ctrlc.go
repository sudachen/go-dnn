package fu

import (
	"os"
	"os/signal"
	"syscall"
)

func WaitForCtrlC() {
	var signal_channel chan os.Signal
	signal_channel = make(chan os.Signal, 1)
	signal.Notify(signal_channel, syscall.SIGINT, syscall.SIGTERM)
	for {
		select {
		case <-signal_channel:
			return
		}
	}
}

