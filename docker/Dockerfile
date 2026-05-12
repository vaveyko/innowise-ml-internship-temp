FROM apache/airflow:3.2.0

COPY --chown=airflow:0 pyproject.toml /opt/airflow
COPY --chown=airflow:0 src/ /opt/airflow/src
COPY --chown=airflow:0 dags/ /opt/airflow/dags

RUN pip install --no-cache-dir .
