# ADR-0005: Keep Corpus Workflow Scheduling External

**Status:** Accepted

## Context

Portfolio corpus generation will eventually run periodically, but scheduling needs
differ across local development, servers, and hosted automation. Scheduling is not
part of corpus transformation itself.

## Decision

Expose `PortfolioCorpusWorkflow` as an idempotent command and keep clocks and
scheduling outside the application. Cron, systemd timers, GitHub Actions, or another
orchestrator may invoke it. A future `--if-changed` mode may stop before LLM work when
the relevant snapshot inputs are unchanged.

## Consequences

- The workflow remains independently testable and portable between schedulers.
- Deployment owns concurrency control, credentials, retries, alerting, and cadence.
- The initial experiment performs manual single-repository runs.
- Scheduled multi-project refresh and automatic index rebuilding remain separate
  implementation work.
