# Aurora Gateway — Operator Handbook

The Aurora Gateway fronts the document API. This handbook covers how an operator
configures and runs one. Quotas and size ceilings live in a separate document, the
*Service Limits and Quotas* sheet; incidents live in the *Incident Runbook*.

## Network and listeners

By default the gateway binds to `0.0.0.0:8443`. Port 8443 is the default because the
listener speaks TLS; a plaintext listener on 8443 is refused at startup rather than
silently downgraded. The bind address is set in the gateway configuration file, which
the installer writes on first run.

## Authentication and tokens

Tokens are short-lived by design. The variable `AURORA_TOKEN_TTL` sets how long a token
remains valid, in seconds, and the shipped default is 3600 seconds — one hour. Setting
it to 0 disables token authentication entirely and is only appropriate on a private
network segment.

## Health and readiness

`/healthz` reports that the process is alive. It answers even when the gateway cannot
reach its database, because a liveness probe that fails on a dependency turns a
recoverable outage into a restart loop. `/readyz` reports that the gateway is ready to
accept traffic and goes unhealthy while the session pool is still filling.

## Runtime limits

The gateway allows 512 concurrent sessions. A request beyond that is queued for up to 20
milliseconds and then rejected; the queue is deliberately short, because a long queue
converts a slow database into an unbounded pile of stuck connections.

## Shutdown

On `SIGTERM` the gateway stops accepting new work and waits 20 seconds for in-flight
requests to finish. After that window it exits anyway. The grace period is not
configurable, because a grace period long enough to matter would exceed the orchestrator
restart deadline.

## Logging

The gateway writes JSON lines to standard error, one object per event. Each object
carries a timestamp, a severity, and the request identifier. Log output is never routed
to a file by the service; the deployment owns that.

## Configuration reload

Sending `SIGHUP` makes the gateway re-read its configuration file and apply the changes
without restarting, which takes effect for new connections only. In-flight requests
finish on the old configuration. There is no reload of credentials, which require a
restart.

## Versioning

Versions are written as `vMAJOR.MINOR.PATCH`, for example `v3.2.1`. A patch release
never changes the meaning of a configuration key. Minor releases may add keys; a release
that removes a key is a major release.
