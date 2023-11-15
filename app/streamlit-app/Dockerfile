FROM python:3.11.4-slim

WORKDIR /work

RUN pip install --upgrade pip

COPY requirements-torch-cpu.txt /tmp/requirements-torch-cpu.txt

RUN pip install --no-cache-dir --requirement /tmp/requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --requirement  /tmp/requirements.txt

ENV PORT=

ENV VESPA_ENDPOINT=

COPY app.py /work/app/app.py

CMD ["sh", "-c", "streamlit run app/app.py --server.port=${PORT} --browser.serverAddress=0.0.0.0 --theme.base=light"]
