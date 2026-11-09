FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ARG WITH_VLA=0
ARG CUROBO_REV=78fd485fa82d9b9a063fb4985e371814587e666a
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_CACHE_DIR=/tmp/uv-cache \
    UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10 \
    PLANNER_PYTHON=/opt/venv-planner/bin/python BIMBO_IK_PYTHON=/opt/venv-planner/bin/python VLA_PYTHON=/opt/venv-vla/bin/python \
    FROZEN_RGBD_PYTHON=/opt/venv-frozen-rgbd/bin/python \
    PORT=8000 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git curl libgl1 libglib2.0-0 \
       build-essential linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/uv

RUN git init -q /opt/curobo && git -C /opt/curobo remote add origin https://github.com/NVlabs/curobo \
    && for i in 1 2 3 4 5; do git -C /opt/curobo fetch -q --depth 1 origin ${CUROBO_REV} && break; sleep 10; done \
    && git -C /opt/curobo checkout -q FETCH_HEAD
RUN --mount=type=cache,target=/tmp/uv-cache uv python install 3.11 && uv venv /opt/venv-planner --python 3.11 \
    && uv pip install --python /opt/venv-planner/bin/python --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/cu128 \
       torch==2.11.0 numpy==2.4.6 scipy==1.17.1 warp-lang==1.12.1 msgpack==1.1.0 websockets==16.1.1 opencv-python-headless==5.0.0.93 scikit-image==0.26.0 "/opt/curobo[cu12-torch]"

COPY solutions/frozen_rgbd/requirements.txt /tmp/frozen-rgbd-requirements.txt
RUN --mount=type=cache,target=/tmp/uv-cache uv venv /opt/venv-frozen-rgbd --python 3.11 \
    && uv pip install --python /opt/venv-frozen-rgbd/bin/python -r /tmp/frozen-rgbd-requirements.txt \
    && uv pip install --python /opt/venv-frozen-rgbd/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu

COPY solutions/larchenko_2025/upstream /opt/submit/solutions/larchenko_2025/upstream
RUN --mount=type=cache,target=/tmp/uv-cache if [ "$WITH_VLA" = "1" ]; then \
      cd /opt/submit/solutions/larchenko_2025/upstream/openpi \
      && GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT=/opt/venv-vla uv sync --frozen --no-dev --python 3.11 \
      && uv pip install --python /opt/venv-vla/bin/python --no-deps -e /opt/submit/solutions/larchenko_2025/upstream \
      && uv pip install --python /opt/venv-vla/bin/python pytest==8.3.5; \
    fi

COPY . /opt/submit
WORKDIR /opt/submit
EXPOSE 8000
CMD ["/opt/venv-planner/bin/python", "-u", "/opt/submit/server/router.py"]
