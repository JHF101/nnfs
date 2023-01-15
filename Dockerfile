# Build all other
FROM python:3.10-slim as base_libs
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# FROM python:3.9-alpine as base_libs
# RUN apk add --no-cache \
#         build-base \
#         gcc \
#         gfortran \
#         openblas-dev \
#         musl-dev \
#         linux-headers

COPY ./library/requirements.txt .
RUN python -m pip install --upgrade pip setuptools
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# FROM python:3.9-alpine as custom_libs
FROM python:3.10-slim as custom_libs
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools
COPY ./library/requirements.txt requirements.txt

COPY --from=base_libs /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt

RUN pip install -e library
# ------------------------------- #
# COPY ./library/nnfs/ nnfs/
# COPY ./library/setup.py setup.py
# RUN python setup.py sdist bdist_wheel
# RUN cp dist/nnfs-0.4.0-py3-none-any.whl /wheels
# ------------------------------- #
