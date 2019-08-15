FROM sudachen/linux:latest
LABEL maintainer="Alexey Sudachen <alexey@sudachen.name>"

USER root
RUN curl -L https://github.com/sudachen/mxnet/releases/download/1.5.0-mkldnn-static/libmxnet_cpu.7z -o /tmp/mxnet.7z \
 && 7z x /tmp/mxnet.7z -o/ \
 && rm /tmp/mxnet.7z \
 && chmod +r -R /opt/mxnet \
 && chmod +x $(find /opt/mxnet -type d) \
 && chmod +x $(find /opt/mxnet/lib -type f) \
 && ln -sf libmxnet_cpu.so /opt/mxnet/lib/libmxnet.so

USER $USER
WORKDIR $HOME


