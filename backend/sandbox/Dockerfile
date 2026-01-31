FROM python:3.11-slim

RUN useradd -m sandbox \
    && pip install --no-cache-dir pandas matplotlib numpy tabulate scikit-learn scipy seaborn statsmodels

USER sandbox
WORKDIR /sandbox
