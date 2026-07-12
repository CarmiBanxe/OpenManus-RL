FROM rocm/pytorch:latest

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-venv \
    git curl wget vim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-legion.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-legion.txt

WORKDIR /app
COPY . /app/
RUN pip install -e .

EXPOSE 8000 8080 8082 5009

CMD ["python", "scripts/start_legion_decision.py"]
