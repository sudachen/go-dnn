null  :=
space := $(null) #
comma := ,

PKGSLIST = mx mx/internal
COVERPKGS= $(subst $(space),$(comma),$(strip $(foreach i,$(PKGSLIST),github.com/sudachen/go-dnn/$(i))))

build:
	cd mx; go build

run-tests:
	#mkdir -p $${GOPATH:-$(HOME)/go}/src/github.com/sudachen
	#ln -sf $$(pwd) $${GOPATH:-$(HOME)/go}/src/github.com/sudachen/go-dnn
	go test -coverprofile=c.out -coverpkg=$(COVERPKGS) ./...

