# Core Eval repository fixtures

Each committed fixture is a directory with exactly two source snapshots:

```text
<fixture-id>/
  base/
  head/
```

`base/` and `head/` are complete repository trees, not patches and not Git
repositories. Ground truth, Suite manifests, evaluator configuration, expected
findings, and clarification answers must remain outside both trees.

`FixtureRepositoryBuilder` turns the two snapshots into a deterministic bare
Git repository. It uses fixed author/committer identities, timestamps and
messages; treats every source file as mode `100644`; sorts UTF-8 paths; and
creates the head commit with the base commit as its parent. File mtimes and the
host checkout's Git configuration do not affect the resulting revisions.

Fixture trees may contain regular files and directories only. Symlinks,
reparse points, special files, VCS metadata components and nested repositories
are rejected. Empty directories are not represented by Git. The fixed
The repository isolation policy also rejects Git symlink entries, LFS
attributes/pointers, non-portable paths and case/NFC path collisions. Gitlinks
and `.gitmodules` are retained as opaque parent-repository metadata; their
submodule repositories are never fetched or materialized. Hooks, filters, and
LFS are never run.

Case authors first run the builder, then copy the returned full base/head
object IDs into the canonical `EvalInput.repository` descriptor. Runtime
preparation rebuilds the fixture and fails closed if either revision or tree
digest differs, so edited fixture bytes cannot silently change an existing
Case.
