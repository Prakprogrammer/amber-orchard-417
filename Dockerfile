# The interpreter image (pinned by digest) and packages the reference labels were made with,
# so the sqlglot tests run here behave exactly like the scorer's.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -r /tmp/constraints.txt

# sqlglot at the pinned commit, byte-compiled so imports while running are cheap
ARG SQLGLOT_COMMIT=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
ADD https://github.com/tobymao/sqlglot/archive/${SQLGLOT_COMMIT}.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /opt/sqlglot \
    && tar xzf /tmp/sqlglot.tar.gz -C /opt/sqlglot --strip-components=1 \
    && rm /tmp/sqlglot.tar.gz \
    && python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests \
    && chmod -R a+rX /opt/sqlglot

COPY ripple /opt/ripple/ripple
COPY atlas /opt/ripple/atlas
WORKDIR /opt/ripple
RUN mkdir -p cache \
    && (python -m ripple.prepare tpc-rows cache/tpc_rows.pkl || echo "no stored TPC rows") \
    && (python -m ripple.prepare dialect-families cache/dialect_families.pkl || echo "no dialect family table") \
    && python -m compileall -q ripple \
    && chmod -R a+rX /opt/ripple

ENV SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["python", "-m", "ripple"]
