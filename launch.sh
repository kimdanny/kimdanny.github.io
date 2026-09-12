#!/usr/bin/env bash
# Build the site, serve it at http://127.0.0.1:4000, and rebuild on every save.
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.rbenv/shims:$PATH"

pkill -f jekyll 2>/dev/null || true    # stop an older server if one is up
sleep 1

LOG=/tmp/jekyll-preview.log
nohup bundle exec jekyll serve --livereload >"$LOG" 2>&1 &

# Wait for the first build, then report whether it actually came up.
for _ in $(seq 20); do
  sleep 1
  if curl -sf -o /dev/null http://127.0.0.1:4000/; then
    echo "Site is up:  http://127.0.0.1:4000"
    echo "Edits rebuild and refresh the browser automatically."
    echo "Stop it with:  bash takedown.sh"
    exit 0
  fi
done

echo "Server failed to start. Log:"
cat "$LOG"
exit 1
