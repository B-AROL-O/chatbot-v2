#!/bin/bash
#
# Run sme-chatbot-eval against a chatbot backend and publish the report as PDF.
# Run with --help for the full documentation.

set -euo pipefail

# --- Known targets -----------------------------------------------------------
# Add new environments here: <name>|<api-url>
LOCALHOST_URL=http://localhost:8000
REMOTE_URL=${EVAL_REMOTE_URL:-}   # set EVAL_REMOTE_URL to use --target remote

# --- Defaults ------------------------------------------------------------------
TARGET=localhost
API_URL=""
RUN_NAME=""
EVAL_DATASET=${EVAL_DATASETS_DIR:-data/eval/datasets}/eval-chatbot-arol-domande.csv
EVAL_DOMAIN=documents
OUT_DIR=${EVAL_RESULTS_DIR:-data/eval/results}
PDF_DIR=data/eval-reports
PUBLISH_PDF=1
VERBOSE=1
EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [OPTIONS] [-- EVAL_ARGS...]

Run the golden question set against a chatbot backend with sme-chatbot-eval,
then convert the resulting report.md into a timestamped PDF.

By default the backend on localhost is evaluated (same behavior as the former
eval_chatbot_localhost.sh).

TARGET SELECTION
  -t, --target NAME     Named backend to evaluate (default: localhost):
                          localhost  $LOCALHOST_URL
                          remote     \$EVAL_REMOTE_URL (currently: ${REMOTE_URL:-unset})
  -u, --api-url URL     Explicit backend base URL. Overrides the URL implied by
                        --target. If --target is not given the run name
                        defaults to "custom".

EVAL OPTIONS
  -d, --dataset CSV     Question set (default: $EVAL_DATASET)
  -D, --domain NAME     Chat domain: sample_cdo | sample_manuals | documents
                        (default: $EVAL_DOMAIN)
  -n, --run-name NAME   Run name; the run directory is
                        <out>/<YYYY-MM-DD>-<run-name>/ (default: the target
                        name, or "custom" with --api-url alone)
  -o, --out DIR         Output root for run records (default: $OUT_DIR)

REPORT OPTIONS
      --pdf-dir DIR     Where the PDF report goes (default: $PDF_DIR)
      --no-pdf          Skip the PDF step; only run the evaluation

OTHER
  -q, --quiet           Do not echo executed commands (default: echo them)
  -h, --help            Show this help and exit

PASSING EXTRA ARGUMENTS TO sme-chatbot-eval
  Everything after a literal "--" is forwarded verbatim to sme-chatbot-eval.
  Useful ones (see "uv run sme-chatbot-eval --help" for all):
    --only 'Q01,Q05-Q08'   run a subset of the questions
    --repeat N             ask each question N times
    --concurrency N        parallel requests (default 1)
    --timeout SECONDS      read timeout (default 180)
    --judge                resolve '?' items with an LLM judge
    --insecure             skip TLS verification
    --dry-run              do not call the backend
    --no-write             do not write a run record (note: the PDF step then
                           has no report.md to convert; combine with --no-pdf)

OUTPUTS
  Run record   <out>/<YYYY-MM-DD>-<run-name>/   (report.md, results, summary)
  PDF report   <pdf-dir>/<YYYY-MM-DD-HHMM>-chatbot-eval-report.pdf
  The PDF is produced by scripts/publish-md-pdf.sh.

EXAMPLES
  # Default: evaluate the local backend, produce the PDF
  $(basename "$0")

  # Evaluate a deployment
  EVAL_REMOTE_URL=http://deploy-host:8000 $(basename "$0") --target remote

  # Quick smoke test of 3 questions on that deployment, no PDF
  $(basename "$0") -t remote --no-pdf -- --only 'Q01-Q03'

  # Arbitrary backend, custom run name and dataset
  $(basename "$0") -u http://10.0.0.5:8000 -n staging -d data/eval/datasets/other-domande.csv

  # Parallel run with LLM judge on the local backend
  $(basename "$0") -- --concurrency 4 --judge

NOTES
  - Run from anywhere: the script changes to the repository root first.
  - Requires uv and, for the PDF step, whatever scripts/publish-md-pdf.sh needs.
  - The report path assumes the run finishes on the same calendar day it
    started (the date is evaluated when the PDF step runs).
USAGE
}

die() { echo "error: $*" >&2; echo "Try '$(basename "$0") --help'." >&2; exit 2; }

need_arg() { [[ $# -ge 2 && -n "$2" ]] || die "option $1 requires a value"; }

# --- Parse arguments -------------------------------------------------------------
TARGET_GIVEN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--target)   need_arg "$@"; TARGET=$2; TARGET_GIVEN=1; shift 2 ;;
    -u|--api-url)  need_arg "$@"; API_URL=$2; shift 2 ;;
    -d|--dataset)  need_arg "$@"; EVAL_DATASET=$2; shift 2 ;;
    -D|--domain)   need_arg "$@"; EVAL_DOMAIN=$2; shift 2 ;;
    -n|--run-name) need_arg "$@"; RUN_NAME=$2; shift 2 ;;
    -o|--out)      need_arg "$@"; OUT_DIR=$2; shift 2 ;;
    --pdf-dir)     need_arg "$@"; PDF_DIR=$2; shift 2 ;;
    --no-pdf)      PUBLISH_PDF=0; shift ;;
    -q|--quiet)    VERBOSE=0; shift ;;
    -h|--help)     usage; exit 0 ;;
    --)            shift; EXTRA_ARGS=("$@"); break ;;
    *)             die "unknown option: $1" ;;
  esac
done

# --- Resolve target ----------------------------------------------------------------
if [[ -z "$API_URL" ]]; then
  case "$TARGET" in
    localhost) API_URL=$LOCALHOST_URL ;;
    remote)    API_URL=$REMOTE_URL
               [[ -n "$API_URL" ]] || die "--target remote needs EVAL_REMOTE_URL" ;;
    *)         die "unknown target '$TARGET' (expected: localhost, remote; or use --api-url)" ;;
  esac
elif [[ $TARGET_GIVEN -eq 0 ]]; then
  TARGET=custom
fi
[[ -n "$RUN_NAME" ]] || RUN_NAME=$TARGET

cd "$(dirname "$0")/.."
[[ -f "$EVAL_DATASET" ]] || die "dataset not found: $EVAL_DATASET"

[[ $VERBOSE -eq 1 ]] && set -x

# --- Run the evaluation --------------------------------------------------------------
uv run sme-chatbot-eval \
  --dataset "$EVAL_DATASET" \
  --api-url "$API_URL" \
  --domain "$EVAL_DOMAIN" \
  --run-name "$RUN_NAME" \
  --out "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"

# --- Publish the report -----------------------------------------------------------------
if [[ $PUBLISH_PDF -eq 1 ]]; then
  scripts/publish-md-pdf.sh \
    --output-dir "$PDF_DIR" \
    --output-name "$(date +%Y-%m-%d-%H%M)-chatbot-eval-report.pdf" \
    "$OUT_DIR/$(date +%Y-%m-%d)-$RUN_NAME/report.md"
fi

# EOF
