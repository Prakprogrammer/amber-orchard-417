# The image the atlas is built in: same interpreter and packages as the runtime image, the pinned
# sqlglot checkout, and this repository at /repo (mounted). For example:
#   docker build -f forge/Dockerfile -t ripple-forge .
#   docker run --rm -v $PWD:/repo -v $PWD/out:/out ripple-forge python -m forge.import_trace /out/import_trace.json /data/tests.json
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -r /tmp/constraints.txt
ARG SQLGLOT_COMMIT=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
ADD https://github.com/tobymao/sqlglot/archive/${SQLGLOT_COMMIT}.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /opt/sqlglot && tar xzf /tmp/sqlglot.tar.gz -C /opt/sqlglot --strip-components=1 && rm /tmp/sqlglot.tar.gz
# forge/ on the path too: its sitecustomize.py records coverage in spawned child processes
ENV PYTHONPATH=/repo:/repo/forge RIPPLE_SQLGLOT=/opt/sqlglot SKIP_INTEGRATION=0
WORKDIR /repo
