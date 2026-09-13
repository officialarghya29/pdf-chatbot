#!/usr/bin/env bash
# Starts backend (:8000) and frontend (:5173) together. Ctrl+C stops both.
set -e
cd "$(dirname "$0")"

if [ ! -f backend/.env ]; then
  echo "backend/.env not found. Copy backend/.env.example and set OPENAI_API_KEY first."
  exit 1
fi

trap 'kill 0' EXIT
echo "- starting backend on :8000"
(cd backend && venv/bin/uvicorn main:app --reload --port 8000 &> /tmp/unfold-backend.log) &
echo "- starting frontend on :5173"
(cd frontend && npm run dev &> /tmp/unfold-frontend.log) &
wait
