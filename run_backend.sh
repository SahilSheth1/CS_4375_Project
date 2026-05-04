#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

python src/data_setup.py

cd backend
uvicorn main:app --reload --port 8000
