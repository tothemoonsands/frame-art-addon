FROM alpine:3.20

WORKDIR /app

RUN apk add --no-cache \
  python3 \
  py3-pip \
  py3-setuptools \
  py3-wheel \
  git \
  ca-certificates \
  py3-pillow \
  zlib \
  libjpeg-turbo \
  freetype

# Allow pip installs in Alpine system python (PEP 668)
RUN python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir --break-system-packages \
  requests \
  "samsungtvws[async] @ git+https://github.com/NickWaterton/samsung-tv-ws-api.git"

COPY run.sh /app/run.sh
COPY uploader.py /app/uploader.py
COPY cover_art.py /app/cover_art.py

RUN chmod +x /app/run.sh

CMD ["/bin/sh", "/app/run.sh"]