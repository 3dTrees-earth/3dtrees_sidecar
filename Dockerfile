FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY src/ /src/

# No ENTRYPOINT: Galaxy runs the job script with /bin/sh inside the container.
WORKDIR /data
