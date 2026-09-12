#!/usr/bin/env bash
# Stop the local preview server.
cd "$(dirname "$0")"

if pkill -f jekyll 2>/dev/null; then
  echo "Server stopped."
else
  echo "No server was running."
fi
