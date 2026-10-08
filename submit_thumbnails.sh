#!/bin/bash
# Submit one SLURM array task per data folder (FAS RC Cannon; edit the #SBATCH lines for other clusters).
#   bash submit_thumbnails.sh /path/to/data/20250127_experiment [FOLDER ...] [-- extra make_thumbnails.py args]
#   bash submit_thumbnails.sh /path/to/data/2025*      # many folders at once
# Outputs default to <folder>/_thumbnails/ (see make_thumbnails.py --out).
# Python: $THUMB_PYTHON if set, else the `python` on PATH when you submit (activate the env first).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

FOLDERS=(); EXTRA=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
    FOLDERS+=("$(realpath -s "$1")"); shift
done
[ ${#FOLDERS[@]} -gt 0 ] || { sed -n '2,7p' "$0"; exit 1; }

PYTHON=${THUMB_PYTHON:-$(command -v python || true)}
if ! env -u PYTHONPATH "${PYTHON:-python}" -c 'import tifffile, zarr, imageio_ffmpeg, PIL' 2>/dev/null; then
    echo "Python '${PYTHON:-<none>}' lacks tifffile/zarr/imageio-ffmpeg/pillow." >&2
    echo "Activate the env from environment.yml, or set THUMB_PYTHON=/path/to/env/bin/python" >&2
    exit 1
fi

mkdir -p "$HERE/logs"
LIST="$HERE/logs/folders_$(date +%Y%m%d_%H%M%S).txt"
printf '%s\n' "${FOLDERS[@]}" > "$LIST"

sbatch --array=1-${#FOLDERS[@]}%10 <<EOT
#!/bin/bash
#SBATCH -J thumbnails
#SBATCH -p shared,sapphire
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-08:00:00
#SBATCH -o $HERE/logs/thumb-%A_%a.out
#SBATCH --open-mode=append

unset PYTHONPATH
FOLDER=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$LIST")
"$PYTHON" "$HERE/make_thumbnails.py" --threads \$SLURM_CPUS_PER_TASK "\$FOLDER" ${EXTRA[@]+"${EXTRA[@]}"}
EOT
