# Build llama.cpp with Unsloth's GLM-5.3-Flash branch on DGX Spark (GB10 / SM 12.1)
# Requires NVIDIA Container Toolkit with --gpus all at runtime.
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ARG LLAMA_REPO=https://github.com/unslothai/llama.cpp
ARG LLAMA_BRANCH=glm5next/upstream

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libcurl4-openssl-dev \
    pciutils \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --branch "${LLAMA_BRANCH}" "${LLAMA_REPO}" llama.cpp \
  && cmake llama.cpp -B llama.cpp/build \
       -DBUILD_SHARED_LIBS=OFF \
       -DGGML_CUDA=ON \
       -DCMAKE_CUDA_ARCHITECTURES="121" \
  && cmake --build llama.cpp/build --config Release -j --clean-first \
       --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

WORKDIR /workspace
EXPOSE 8000

ENTRYPOINT ["/opt/llama.cpp/build/bin/llama-server"]
