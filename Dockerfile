FROM python:3.9-slim

# Install the nnfs-lib
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip setuptools
RUN python -m pip install -r requirements.txt
RUN python setup.py sdist bdist_wheel
RUN pip install dist/nnfs-0.4.0-py3-none-any.whl
WORKDIR /app

EXPOSE 8501
# RUN streamlit run app.py
# ENTRYPOINT ["streamlit", "run"]
# CMD ["app.py"]
CMD ["streamlit" "run" "--server.port" "8501" "main.py"]