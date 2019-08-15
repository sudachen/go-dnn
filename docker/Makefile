
%.Build: %.Dockerfile
	docker build -t sudachen/$(basename $@) -f $^ .

%.Push : %.Build
	docker push sudachen/$(basename $@)

ALL = mxnet go-mxnet-ci

build: $(foreach i,$(ALL),$(basename $(i)).Build)
push: $(foreach i,$(ALL),$(basename $(i)).Push)

