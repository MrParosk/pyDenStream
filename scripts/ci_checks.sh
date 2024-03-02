#!/bin/bash

set -e

echo 'running ruff'
ruff check .

echo 'running isort'
isort . --check

echo 'running black'
black . --check

echo 'running mypy'
mypy .
