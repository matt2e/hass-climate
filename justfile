test *args:
    .venv/bin/python -m pytest tests/ {{args}}

check:
    ruff format --check .
    ruff check .

fmt:
    ruff format .
    ruff check --fix .
