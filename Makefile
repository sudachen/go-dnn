null  :=
space := $(null) #
comma := ,

PKGSLIST = mx mx/internal
COVERPKGS= $(subst $(space),$(comma),$(strip $(foreach i,$(PKGSLIST),github.com/sudachen/go-mxnet/$(i))))

build:
	cd mx; go build

run-tests:
	cd tests && go test -coverprofile=../c.out -coverpkg=$(COVERPKGS)

