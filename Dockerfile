FROM python:3.9-slim as base
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY ./requirements.txt /
RUN python -m pip install --upgrade pip setuptools
RUN ls
COPY nnfs/ nnfs/
COPY setup.py setup.py
RUN pip wheel --wheel-dir /wheels -r requirements.txt
RUN python setup.py sdist bdist_wheel
RUN cp dist/nnfs-0.4.0-py3-none-any.whl /wheels

FROM python:3.9-slim
COPY --from=base /wheels /wheels
RUN python -m pip install --upgrade pip
RUN pip install --no-cache /wheels/*
WORKDIR /app
COPY . .
EXPOSE 8501
# RUN streamlit run app.py
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]