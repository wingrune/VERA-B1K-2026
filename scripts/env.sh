# Source this, then `load_env <file>`: exports KEY=VALUE lines of a .env file without executing it.
# Variables already set in the environment win; blank lines and # comments are skipped; surrounding quotes are removed.
load_env() {
  [ -f "$1" ] || return 0
  local line key val
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    key=${line%%=*}; val=${line#*=}
    key=${key#export }; key=${key// /}
    [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    val=${val%\"}; val=${val#\"}; val=${val%\'}; val=${val#\'}
    [ -n "${!key:-}" ] || export "$key=$val"
  done < "$1"
}
