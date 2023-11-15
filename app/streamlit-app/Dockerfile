FROM python:3.11.4-slim
LABEL maintainer="Filippo Broggini <https://github.com/filippo82>"

WORKDIR /work

RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --requirement  /tmp/requirements.txt

ENV PORT=

ENV VESPA_ENDPOINT=

COPY app.py /work/app/app.py

CMD ["sh", "-c", "streamlit run app/app.py --server.port=${PORT} --browser.serverAddress=0.0.0.0 --theme.base=light"]
