FROM node:22-bookworm

# Fermyon Spin CLI (v3, matching mise.toml). The installer drops the binary in
# the working dir; move it onto PATH.
# Must be a full release tag (e.g. v3.6.3) — the installer plugs this straight
# into the download URL, so a bare "v3" 404s.
ARG SPIN_VERSION=v3.6.3
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash -s -- -v ${SPIN_VERSION} \
  && mv ./spin /usr/local/bin/spin

WORKDIR /app

# Install deps first so the layer caches across source-only changes.
COPY package.json package-lock.json ./
RUN npm install

# Build the component (esbuild -> j2w -> target/few-shot.wasm).
COPY . .
RUN spin build

EXPOSE 3000

# Spin reads variables from SPIN_VARIABLE_<NAME> in the environment. These are
# placeholders, not real config — override every one at `docker run` time with
# -e SPIN_VARIABLE_<NAME>="...". Note: an ENV set here overrides the spin.toml
# default, so vars with meaningful defaults are seeded with that same default.
# These are placeholders and not actual values
ENV SPIN_VARIABLE_VALKEY_URL="rediss://CHANGEME"
ENV SPIN_VARIABLE_INFERENCE_API_URL="http://localhost:11434/v1/chat/completions"
ENV SPIN_VARIABLE_CHAT_MODEL="gpt-oss:20b"
ENV SPIN_VARIABLE_INFERENCE_API_KEY=""
ENV SPIN_VARIABLE_CACHE_THRESHOLD=0.95
ENV SPIN_VARIABLE_CACHE_TTL_SECONDS=3600

# --log-dir "" disables Spin's default file logging (.spin/logs/) so component
# stdout/stderr go straight to the container's stdout/stderr, where `docker logs`
# and the deploy platform can capture them. Without this, component errors are
# written to files inside the container and are invisible on a deployed instance.
CMD ["spin", "up", "--log-dir", "", "--listen", "0.0.0.0:3000"]
