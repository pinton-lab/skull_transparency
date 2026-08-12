# skull-transparency: compute + explore transparency maps in a container.
# The CUDA solver binary is NOT baked in -- `compute` fetches it on first use and
# shows its license -- so this image is Apache-2.0-clean. GPU access comes from the
# host driver via `docker run --gpus all` (the solver is statically linked).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# host driver injection (NVIDIA container toolkit)
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY . /opt/skull_transparency
RUN pip install --no-cache-dir '/opt/skull_transparency[compute]' matplotlib

# cache solver/skull/gallery downloads on a mountable volume
ENV SKULL_TRANSPARENCY_CACHE=/cache
VOLUME /cache

ENTRYPOINT ["skull-transparency"]
CMD ["--help"]
