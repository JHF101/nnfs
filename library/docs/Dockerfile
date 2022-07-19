# FROM python:3.10-alpine as base
# RUN apk upgrade
# RUN apk add gcc build-base musl-dev

FROM python:3.10-slim as base
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel setuptools

WORKDIR /wheels
COPY ../../library/requirements.txt requirements.txt
RUN python -m pip wheel sphinx sphinx_rtd_theme
RUN python -m pip wheel --wheel-dir /wheels -r requirements.txt

FROM base
RUN pip install --upgrade pip wheel setuptools
COPY --from=base /wheels/*.whl /wheels/
RUN python -m pip install *.whl

WORKDIR /library
COPY ../../library /library

CMD [ "/bin/sh" ]