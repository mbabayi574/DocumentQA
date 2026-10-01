#!/usr/bin/env bash
# End-to-end smoke test against a running server (plan.md P8 Gate).
#
# add -> query -> edit -> query -> delete -> query, asserting the *content* at each step. The
# point is the transitions, not the endpoints: an edit must make the old text unreachable and
# the new text reachable, and a delete must make both unreachable. A smoke test that only
# checks "200 OK" would pass against a system that never updated anything.
#
# Usage: scripts/smoke_test.sh [base_url]     (default http://127.0.0.1:8000)
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

V1='# Handbook

## Install

Run the installer on Linux. It checks the kernel version first.

## Usage

The service returns ERR-404 when the token is invalid.
'
V2='# Handbook

## Install

Run the installer on Linux. It checks the kernel version first.

## Usage

The service returns ERR-503 when the upstream cache is unreachable.
'

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; exit 1; }
step() { printf '\n== %s\n' "$1"; }

# field <json> <python-expression over `d`>
field() { printf '%s' "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print($2)"; }
contains() { case "$1" in *"$2"*) return 0 ;; *) return 1 ;; esac; }
# `curl -f` exits non-zero on a 404, which `set -e` treats as a script failure -- but a 404 is
# the expected answer in several steps below. So no `-f`: branch on the status code instead.
status_of() { curl -sS -o /dev/null -w '%{http_code}' "$@"; }

printf 'Smoke test against %s\n' "$BASE"

step "health and readiness"
HEALTH="$(curl -fsS "$BASE/health")"
contains "$HEALTH" '"ok"' || fail "health did not report ok: $HEALTH"
pass "health"

READY="$(curl -fsS "$BASE/ready")"
contains "$READY" '"ready"' || fail "not ready: $READY"
pass "ready (model=$(field "$READY" 'd["model_id"]'), thresholds_calibrated=$(field "$READY" 'd["thresholds_calibrated"]'))"

step "add"
# Establish the precondition rather than assume it. There is deliberately no delete-all
# endpoint, so the only way to start from a known state is to remove our own document first --
# and a POST into a live store with different content is a 409 by design, which is exactly the
# kind of thing a smoke test should not trip over when re-run.
PREV="$(status_of "$BASE/documents/handbook")"
if [ "$PREV" = "200" ]; then
  status_of -X DELETE "$BASE/documents/handbook" > /dev/null
  pass "removed a pre-existing handbook"
fi
printf '%s' "$V1" > "$TMP/handbook.md"
CREATED="$(curl -fsS -X POST "$BASE/documents" -F "file=@$TMP/handbook.md")"
STATUS="$(field "$CREATED" 'd["status"]')"
[ "$STATUS" = "created" ] || fail "expected created, got $STATUS: $CREATED"
CHUNKS="$(field "$CREATED" 'd["chunks_added"]')"
[ "$CHUNKS" -gt 0 ] || fail "no chunks were indexed"
pass "created, $CHUNKS chunks, $(field "$CREATED" 'd["embed_requests"]') embedding request(s)"

step "query before the edit"
A1="$(curl -fsS -X POST "$BASE/query" -H 'content-type: application/json' \
  -d '{"question":"what does the service return when the token is invalid"}')"
contains "$A1" 'ERR-404' || fail "the first version was not answerable: $A1"
contains "$A1" '"answered"' || fail "expected answered: $A1"
pass "answered from v$(field "$A1" 'd["citations"][0]["doc_version"]') with a citation"

step "identical re-upload is unchanged"
SAME="$(curl -fsS -X POST "$BASE/documents" -F "file=@$TMP/handbook.md")"
[ "$(field "$SAME" 'd["status"]')" = "unchanged" ] || fail "re-upload was not unchanged: $SAME"
[ "$(field "$SAME" 'd["embed_requests"]')" = "0" ] || fail "re-upload spent embedding requests"
pass "unchanged, zero embedding requests (I4, I5)"

step "edit"
V_BEFORE="$(field "$CREATED" 'd["version"]')"
printf '%s' "$V2" > "$TMP/handbook.md"
UPDATED="$(curl -fsS -X PUT "$BASE/documents/handbook" -F "file=@$TMP/handbook.md")"
[ "$(field "$UPDATED" 'd["status"]')" = "updated" ] || fail "edit was not applied: $UPDATED"
# Versions are monotonic and never reused, so this must advance by exactly one -- not equal 2.
# A smoke test run against an existing store would otherwise fail for no reason.
V_AFTER="$(field "$UPDATED" 'd["version"]')"
[ "$V_AFTER" -eq "$((V_BEFORE + 1))" ] || fail "version went $V_BEFORE -> $V_AFTER, expected +1"
pass "updated to v$V_AFTER"

step "query after the edit: old text unreachable, new text reachable"
A2="$(curl -fsS -X POST "$BASE/query" -H 'content-type: application/json' \
  -d '{"question":"what does the service return when the token is invalid"}')"
if contains "$A2" 'ERR-404'; then
  fail "superseded text is still reachable (I1): $A2"
fi
pass "the superseded ERR-404 is gone"

A3="$(curl -fsS -X POST "$BASE/query" -H 'content-type: application/json' \
  -d '{"question":"what does the service return when the upstream cache is unreachable"}')"
contains "$A3" 'ERR-503' || fail "the edited text is not reachable: $A3"
pass "the new ERR-503 is answerable"

step "insufficient information carries no citations"
A4="$(curl -fsS -X POST "$BASE/query" -H 'content-type: application/json' \
  -d '{"question":"what is the population of Reykjavik in winter"}')"
[ "$(field "$A4" 'd["status"]')" = "insufficient_information" ] || fail "answered an off-topic question: $A4"
[ "$(field "$A4" 'd["evidence_score"]')" = "0.0" ] || fail "a refusal carried evidence: $A4"
[ "$(field "$A4" 'len(d["citations"])')" = "0" ] || fail "a refusal carried citations"
pass "refused, no citations"

step "delete"
CODE="$(status_of -X DELETE "$BASE/documents/handbook")"
[ "$CODE" = "204" ] || fail "delete returned $CODE, expected 204"
CODE2="$(status_of -X DELETE "$BASE/documents/handbook")"
[ "$CODE2" = "404" ] || fail "a repeated delete returned $CODE2, expected 404"
pass "204, then 404 on the repeat"

step "query after the delete"
A5="$(curl -fsS -X POST "$BASE/query" -H 'content-type: application/json' \
  -d '{"question":"kernel version"}')"
[ "$(field "$A5" 'd["status"]')" = "insufficient_information" ] || fail "deleted content is reachable: $A5"
pass "unreachable (I2)"

step "errors use one envelope"
ERR_CODE="$(status_of "$BASE/documents/definitely-missing")"
[ "$ERR_CODE" = "404" ] || fail "a missing document returned $ERR_CODE, expected 404"
ERR="$(curl -sS "$BASE/documents/definitely-missing")"
contains "$ERR" '"DOCUMENT_NOT_FOUND"' || fail "missing document did not 404 correctly: $ERR"
contains "$ERR" '"request_id"' || fail "the error envelope has no request_id: $ERR"
pass "404 in the standard envelope"

printf '\nSmoke test passed.\n'