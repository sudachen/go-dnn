package mx

import (
	"github.com/sudachen/go-dnn/mx/internal"
)

const (
	VersionMajor = 1
	VersionMinor = 5
	VersionPatch = 0
)

const Version VersionType = VersionMajor*10000 + VersionMinor*100 + VersionPatch

func LibVersion() VersionType {
	return VersionType(internal.LibVersion)
}

func GpuCount() int {
	return internal.GpuCount
}

