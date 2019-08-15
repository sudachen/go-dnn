null  :=
space := $(null) #
comma := ,

PKGSLIST = mx mx/internal
COVERPKGS= $(subst $(space),$(comma),$(strip $(foreach i,$(PKGSLIST),github.com/sudachen/go-dnn/$(i))))

build:
	cd mx; go build

run-tests:
	mkdir -p github.com/sudachen
	ln -sf $$(pwd) github.com/sudachen/go-dnn
	go test -coverprofile=c.out -coverpkg=$(COVERPKGS) ./...
	sed -i -e 's:github.com/sudachen/go-dnn/::g' c.out

