FROM ghcr.io/prefix-dev/pixi:0.53.0-bookworm-slim

# Install only essential system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Set working directory
WORKDIR /app

# Note: Files are mounted via devcontainer volume mount, not copied

# Set the default command to start a bash shell
CMD ["/bin/bash"]
