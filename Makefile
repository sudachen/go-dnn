null  :=
space := $(null) #
comma := ,

PKGSLIST = mx mx/internal nn
COVERPKGS= $(subst $(space),$(comma),$(strip $(foreach i,$(PKGSLIST),github.com/sudachen/go-dnn/$(i))))

build:
	cd mx; go build

run-tests:
	mkdir -p github.com/sudachen
	ln -sf $$(pwd) github.com/sudachen/go-dnn
	cd tests && go test -o ../tests.test -c -covermode=atomic -coverprofile=c.out -coverpkg=../...
	./tests.test -test.v=true -test.coverprofile=c.out
	cp c.out gocov.txt
	sed -i -e 's:github.com/sudachen/go-dnn/::g' c.out
	rm github.com/sudachen/go-dnn

run-cover:
	go tool cover -html=gocov.txt

run-cover-tests: run-tests run-cover


