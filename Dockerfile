# mlarchive container image.
#
# The CUDA libraries Whisper needs (cuBLAS, cuDNN, NVRTC) come from the nvidia-*
# pip wheels in requirements.txt, so a plain Python base is enough; the GPU
# driver itself is provided by the host (Docker Desktop/WSL2 or the NVIDIA
# Container Toolkit) when the container is started with GPU access.

FROM python:3.13-slim AS runtime

# ca-certificates: the system trust store that truststore verifies TLS against.
# house.mi.gov omits its intermediate certificate. Windows fetches it on the fly;
# Linux/OpenSSL doesn't, so we add DigiCert's intermediate (signed by DigiCert
# Global Root G3, valid to 2031) to the store. The chain is still fully verified.
COPY docker/certs/*.crt /usr/local/share/ca-certificates/
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlarchive/ mlarchive/

# Linux counterpart of transcriber._register_cuda_dlls (which only runs on
# Windows): tell the loader where the pip-installed CUDA libraries live.
ENV PYTHONUNBUFFERED=1 \
    MLA_DATA_DIR=/data \
    HF_HOME=/models \
    LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib

VOLUME ["/data", "/models"]
ENTRYPOINT ["python", "-m", "mlarchive"]
CMD ["run"]


# Same image plus the test suite: docker compose --profile test run --rm test
FROM runtime AS test
COPY requirements-dev.txt pytest.ini ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests/ tests/
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-q"]
