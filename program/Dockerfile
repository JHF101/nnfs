# Build all other
FROM python:3.10-slim as base_libs
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# FROM python:3.10-alpine as base_libs
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

# FROM python:3.10-alpine
FROM python:3.10-slim
RUN python -m pip install --upgrade pip

# Program container specific
COPY ./program/app-requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

# Copying all wheels built in first image
COPY --from=base_libs /wheels /wheels
RUN pip install --no-cache /wheels/*.whl

# Copying only custom lib to build
COPY ./library /library
RUN pip install -e library

WORKDIR /app
COPY ./program/*.py .

EXPOSE 8501
# EXPOSE 5000

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
