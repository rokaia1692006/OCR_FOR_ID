#!/bin/sh
set -e

PORT="${PORT:-8080}"
ENABLE_TUNNEL="${ENABLE_TUNNEL:-1}"

echo "Starting uvicorn on port ${PORT}..."
uvicorn fileFinal:app --host 0.0.0.0 --port "${PORT}" &
UVICORN_PID=$!

if [ "$ENABLE_TUNNEL" = "1" ]; then
  echo "Starting Cloudflare Quick Tunnel..."
  cloudflared tunnel --url "http://localhost:${PORT}" --no-autoupdate &
  TUNNEL_PID=$!
  trap 'kill $UVICORN_PID $TUNNEL_PID 2>/dev/null' TERM INT
else
  trap 'kill $UVICORN_PID 2>/dev/null' TERM INT
fi

wait $UVICORN_PID
