#!/bin/bash
# Seed the voices volume with default voices if they don't already exist
for f in /default-voices/*.pt; do
    [ -f "$f" ] || continue
    dest="/.neutts_server/voices/$(basename "$f")"
    [ -f "$dest" ] || cp "$f" "$dest"
done

exec python -m neutts_server "$@"
