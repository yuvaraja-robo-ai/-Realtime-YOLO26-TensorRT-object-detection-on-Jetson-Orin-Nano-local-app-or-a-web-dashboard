#!/usr/bin/env bash
# Orin YOLO detector — process control.
#
#   ./run.sh start [web|local]   start in background (default web), record PID
#   ./run.sh stop                stop the app, release GPU + camera
#   ./run.sh restart [web|local] stop then start
#   ./run.sh status              running? + app PIDs + camera holder
#   ./run.sh web                 run web server in the foreground (Ctrl-C quits)
#   ./run.sh local               run native window in the foreground
#
# PORT=9000 ./run.sh start       web on a different port
#
# Note: no `set -e` on purpose — stop() must run every cleanup step even when
# an individual kill/pgrep returns non-zero (e.g. nothing matched).
cd "$(dirname "$0")" || exit 1

PORT="${PORT:-8000}"
PIDFILE="/tmp/orin-yolo.pid"
LOGFILE="/tmp/orin-yolo.log"
CAM_DEV="${CAM_DEV:-/dev/video0}"

# Matches every app process however launched; does NOT match this script
# (whose command line is "run.sh ...").
APP_PATTERNS='uvicorn app\.main:app|app\.local_viewer'

_app_pids() { pgrep -f "$APP_PATTERNS" 2>/dev/null; }

_running_pid() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
  rm -f "$PIDFILE"; return 1
}

start() {
  local mode="${1:-web}"
  local pid
  if pid="$(_running_pid)"; then
    echo "already running (PID $pid). Use './run.sh restart' or stop first."
    return 1
  fi
  if [ -n "$(_app_pids)" ]; then
    echo "an app process is already running: $(_app_pids). Run './run.sh stop' first."
    return 1
  fi
  fuser -k "$CAM_DEV" 2>/dev/null   # clear any stale camera holder
  sleep 1
  echo "starting '$mode' in background -> $LOGFILE"
  if [ "$mode" = "local" ]; then
    nohup python3 -m app.local_viewer >"$LOGFILE" 2>&1 </dev/null &
  else
    nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 \
      >"$LOGFILE" 2>&1 </dev/null &
  fi
  echo "$!" >"$PIDFILE"     # $! is the python process itself (no setsid wrapper)
  echo "PID $! · web: http://$(hostname -I | awk '{print $1}'):$PORT · log: $LOGFILE"
}

stop() {
  local left
  # 1) Graceful TERM to every app process (recorded PID + any orphan/foreground).
  if [ -n "$(_app_pids)" ]; then
    echo "stopping: $(_app_pids | tr '\n' ' ')"
    pkill -TERM -f "$APP_PATTERNS" 2>/dev/null
  else
    echo "no app process running."
  fi

  # 2) Wait up to ~5s for clean exit (releases CUDA/GPU context + camera).
  for _ in $(seq 1 17); do
    [ -z "$(_app_pids)" ] && break
    sleep 0.3
  done

  # 3) Still alive => ignored SIGTERM (stuck in CUDA/cap.read). Force SIGKILL.
  left="$(_app_pids)"
  if [ -n "$left" ]; then
    echo "forcing SIGKILL: $(echo "$left" | tr '\n' ' ')"
    pkill -9 -f "$APP_PATTERNS" 2>/dev/null
    sleep 1
  fi

  rm -f "$PIDFILE"
  fuser -k "$CAM_DEV" 2>/dev/null   # free camera even if a child still held it

  left="$(_app_pids)"
  if [ -n "$left" ]; then
    echo "WARNING: still alive after SIGKILL: $left"
    return 1
  fi
  echo "stopped — GPU + camera released."
}

status() {
  local pid
  if pid="$(_running_pid)"; then
    echo "RUNNING (PID $pid)"
  else
    echo "STOPPED (no PID file)"
  fi
  echo "app processes: $(_app_pids | tr '\n' ' ' | sed 's/ $//' || true)"
  echo -n "camera $CAM_DEV: "
  if fuser "$CAM_DEV" >/dev/null 2>&1; then echo "held by $(fuser "$CAM_DEV" 2>/dev/null)"; else echo "free"; fi
}

case "${1:-}" in
  start)   shift; start "$@" ;;
  stop)    stop ;;
  restart) shift; stop; sleep 1; start "$@" ;;
  status)  status ;;
  local)   fuser -k "$CAM_DEV" 2>/dev/null; sleep 1; exec python3 -m app.local_viewer "${@:2}" ;;
  web|"")  fuser -k "$CAM_DEV" 2>/dev/null; sleep 1
           exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 ;;
  *)       echo "usage: ./run.sh {start|stop|restart|status|web|local} [web|local]" >&2; exit 1 ;;
esac
