#!/usr/bin/env bash
# Run from a regular terminal after closing Codex.
set -euo pipefail
umask 077

persistent=/NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/codex-persistent
source_home=${CODEX_HOME:-"$HOME/.codex"}
source_binary=$(command -v codex || true)

if [[ ! -d "$source_home" ]]; then
    printf 'Codex state directory does not exist: %s\n' "$source_home" >&2
    exit 1
fi
if [[ -z "$source_binary" || ! -x "$source_binary" ]]; then
    printf 'Cannot find the codex executable in PATH.\n' >&2
    exit 1
fi
if [[ -e "$persistent" || -L "$persistent" ]]; then
    printf 'Destination already exists; leaving it unchanged: %s\n' "$persistent" >&2
    exit 1
fi

printf 'Run this script after closing all Codex sessions.\n'
printf 'Copying %s to %s/home ...\n' "$source_home" "$persistent"
mkdir -p "$persistent/bin" "$persistent/home"
chmod 700 "$persistent" "$persistent/home"

# Preserve the original installation and copy all local state.
cp -a "$source_home/." "$persistent/home/"
chmod 700 "$persistent/home"
# Keep the CLI beside its package resources and helper executables.
package_current="$persistent/home/packages/standalone/current"
release_name=$(basename -- "$(readlink -f -- "$source_binary" | xargs dirname | xargs dirname)")
if [[ ! -f "$persistent/home/packages/standalone/releases/$release_name/codex-package.json" ]]; then
    printf 'Expected standalone package was not found in the copied state.\n' >&2
    exit 1
fi
ln -sfn "releases/$release_name" "$package_current"

cat > "$persistent/start-codex" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
umask 077
persistent=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CODEX_HOME="$persistent/home"
exec "$persistent/home/packages/standalone/current/bin/codex" "$@"
EOF
chmod 700 "$persistent/start-codex"

"$persistent/start-codex" --version
printf '\nMigration complete. Resume from your project directory with:\n%s resume --last\n' \
    "$persistent/start-codex"
