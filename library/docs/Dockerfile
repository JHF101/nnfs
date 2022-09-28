# FROM python:3.10-alpine as base
# RUN apk upgrade
# RUN apk add gcc build-base musl-dev

FROM python:3.10-slim as base
ENV AUTHOR_NAME=AUTHOR_NAME
ARG VERSION=VERSION
ARG CODEBASE_NAME=CODEBASE_NAME

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended latexmk && \
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

WORKDIR /library/nnfs/
# Remove unnescsary modules
COPY ./library/docs/remove-modules.sh remove-modules.sh
RUN chmod +x remove-modules.sh
RUN ./remove-modules.sh

WORKDIR /library/docs
# Remove the files to make folder empty
RUN rm -rf *
# Generate boiler plate code
RUN sphinx-quickstart --quiet -p $CODEBASE_NAME -a $AUTHOR_NAME -v $VERSION
COPY ./library/docs/conf.py ./conf.py

WORKDIR /library
# TODO: Generate modules + compile
RUN sphinx-apidoc -o docs nnfs/

WORKDIR /library/docs
COPY ./library/docs/index.rst ./index.rst

# Make html pages
# RUN make html
# Make pdf document
RUN make latexpdf

VOLUME ./library/docs/build
RUN mkdir build
RUN cp -r ./_build/* ./build/


CMD [ "/bin/bash" ]