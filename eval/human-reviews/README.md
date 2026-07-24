# Evaluator-private human review ledger

This directory is the approval source of truth for Core Eval Cases. It is never
copied into a tested Agent workspace.

`records/` is intentionally absent until a real independent human review is
completed. Do not create placeholder, AI-authored, or hand-edited approvals.
Use `eval/authoring/core_human_review.py import`; it verifies the blind packet,
independent response, deterministic comparison, Author receipt, optional
Adjudicator C receipt, current Case/fixture/protocol binding, and then
publishes one immutable record with atomic no-overwrite semantics.

Both the blind-packet output parent and the ledger parent must be a
permission-controlled, non-shared directory. The writer uses create-new,
no-follow where the platform exposes it, rejects stable symlink/reparse and
hard-link conditions, and captures/rechecks the parent directory chain and
file identities throughout publication. Pure Python stdlib on Windows cannot
fully eliminate a malicious same-privilege directory-swap race; filesystem
ACLs and exclusive ownership of these parent directories are therefore part
of the trust boundary.

The CLI can reject obvious machine identities and inconsistent attestations,
but it cannot prove real-world identity or blindness. Reviewed PR and external
organization/audit evidence remain necessary. `annotation.json` is only a
builder projection of a verified record and cannot satisfy the release gate by
itself.

Hashes and internally consistent receipts prove byte binding and replay
consistency only. They do not prove that a signer is human, independent, blind,
or signed at the claimed real-world time. Those claims require protected PR or
organization identity, CODEOWNERS/review policy, external audit evidence, or a
separately managed digital-signature system.

Until every Core Case has a valid record and every Regression Case has its
required real three-trial baseline, the checked-in Case and manifest files are
pending candidates, not an approved release-gating Suite.
