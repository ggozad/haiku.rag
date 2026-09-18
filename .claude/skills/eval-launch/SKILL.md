---
name: eval-launch
description: Launch an evaluation arm, a paired comparison or a corpus build through the haiku.rag evaluations harness. Use before starting any evaluation run on a GPU box, when asked to compare two code versions or configs, when a queued run looks stuck, and when asked how far a run has got or when it will finish.
---

# Launching evaluation arms

Every arm starts through `evaluations queue`. Nothing else starts a run: no
shell script, no `evaluations run` typed into a tmux pane. The queue runs the
checks a person forgets, and it refuses to start an arm that fails one.

Read `reference.md` in the `haiku.rag-evaluations-data` repository first. It
holds what exists right now: machines, endpoints, databases, checkout layout,
where past results live. That repository is private and is cloned on the GPU
box; this file holds only how to do things.

## 1. Before writing anything

1. **Has it been run?** `evaluations arms list --dataset <dataset>` lists
   every registered arm with its status. Then the "Eval results" section of
   the memory index. The answer is often already there.
2. **Which code?** Pin a full commit sha for every side. Verify two shas name
   the same code by tree, not lineage: `git rev-parse A^{tree}` against
   `git rev-parse B^{tree}`. A rebase orphans a sha while the code is
   unchanged, and an amend rewrites the id with the tree intact.
3. **Which layer does the change live in?** A `--skip-db` arm exercises
   search, expansion, the capability and the judge. It cannot see a
   converter, chunker, embedder or storage change. Those need a fresh small
   build on each side and a deterministic diff, not a QA arm.
4. **Never grep coloured git output.** `git --no-pager diff --no-color A B --
   <paths> | grep -E '^[+-][^+-]'`. With colour on, `^[+-]` matches nothing
   and the diff reads as empty.
5. **Name the revision pair in the sentence** whenever you report a diff, and
   never assert a diff you did not run in this session.
6. **Read both configs before writing either file.** `differences` names every
   config key that differs by dotted path, and the comparison is over the
   resolved configuration, not the file text, so two files that read
   differently and resolve the same show no difference at all.
7. **Write the claim down first.** `hypothesis` is one line saying what the
   pair would show, and the file will not load without it once it names a
   comparator. A claim written after the numbers is not a claim.

## 2. The arm file

One YAML file per arm, kept in `arms/` in the data repository. Paths resolve
against the file.

```yaml
name: <dataset>-<what>-<cases>       # unique in the registry, also the run name
dataset: frames
worktree: ~/wt/<name>                # provisioned by the queue when absent
sha: <full or 12-character commit sha>
config: ../configs/<config>.yaml
db: ~/.local/share/haiku.rag/evaluations/dbs/<database>.lancedb
limit: 150                           # or filter_ids: ../smoke/<ids>.txt
flags: [--skip-db, --skip-retrieval]
smoke_ids: ../smoke/<dataset>-<kind>.txt
comparator: <baseline arm>.yaml      # the arm this one is measured against
differences: [sha]                   # every way the two arms differ
decision_rule: McNemar exact on answer_equivalent, two-sided, treated worse at p < 0.05 fails
hypothesis: <the claim this pair tests, one line>
operator: <who>
deadline_hours: <derived, see below>
```

The null pair is two more files that replicate each other: same `dataset`,
`worktree`, `sha`, `config` and `db`, two new names, one naming the other as
its `comparator`, `differences: []`, and the same `decision_rule` and
`hypothesis`. They may run a smaller `limit` than the pair under test, as long
as both of them run the same one, which is how an expensive dataset gets a
noise floor without paying for the full set twice. Its claim is that the instrument is stable,
so a rejection there means the pair cannot resolve the effect the treated arm
is testing, whatever the treated arm's own p-value says.

`deadline_hours` bounds the run and kills it at the limit; the killed arm is
void and a void arm is never paired, so a deadline set too low costs the whole
run. Derive it, never copy it: take a completed row on the same dataset
(`evaluations arms list --dataset <dataset>`, then `arms show <name>` for
`wall_seconds` and `cases`), multiply the seconds per case by your case count,
and double it when arms share an endpoint, since each one slows the others.
Datasets differ by a factor of three in seconds per case, and the same dataset
differs by more than that between models.

`limit: N` is a deterministic prefix, `select(range(N))` over the dataset's
own order after `filter_ids` has been applied. Two arms with equal limits run
exactly the same cases, and a smaller limit is a prefix of a larger one. It
never samples, so a pair may differ in `limit` only when `differences` says so.

Rules the file must satisfy, all enforced by `evaluations preflight`:

- `differences` names every difference from the comparator and nothing that
  does not differ: arm fields by field (`sha`, `limit`, `db`, `filter_ids`,
  `dataset`), flags by option (`--vacuum-interval`), config keys by dotted
  path (`qa.max_searches`). An unnamed difference stops the launch. So does a
  named one that does not exist.
- `flags` may not carry `--config`, `--name`, `--db`, `--limit` or
  `--filter-ids`. The fields above pin them, and a flag would run something
  else than the registry records.
- A pair carries its `decision_rule` before launch, including which
  direction fails. A rule written after the data exists is not a rule.
- A pair needs a null pair beside it: the same code and config run twice on
  the same cases. Identical code takes a different trajectory on most cases
  and flips a few verdicts; without the null pair those flips read as an
  effect. One null replicate is one draw, not a variance estimate.
- The config's embedder must be the database's stored embedder. The preflight
  compares provider, name and dimension.
- The worktree's `.env` must carry `LOGFIRE_TOKEN`, unless `flags` carries
  `--no-telemetry` and the result file is the record. Without either a run
  exits 0 having recorded nothing, and `evaluations run` refuses to start.
  `--no-telemetry` is refused for a live dataset or `--skip-qa`: only a QA run
  over a static corpus writes a result file.
- An arm whose config fills `lancedb.databases` evaluates that set: it carries
  `--skip-db` and no `db`. Population writes to a database such a run does not
  read.

## 3. The smoke set

`smoke_ids` is a file of case ids, kept in the data repository's `smoke/`. Ten cases, chosen, not
sampled: at least three must be of the kind the change is supposed to affect.
A random ten from several hundred has a fair chance of holding none of a
one-in-eight kind, so the smoke would pass while testing nothing.

The queue runs the smoke first. The full arm starts only when the smoke exits
cleanly and records at least one case. The precondition you check by hand: at least one of the
affected cases resolves its answer from a tool result. Read the tool output in
Logfire (`execute_tool` spans under the case), not the answer text. A model
can answer from memory, so the answer alone does not show the corpus supplied
anything.

## 4. Launching

```bash
cd ~/haiku.rag-evaluations-data && git pull
evaluations queue arms/<baseline>.yaml arms/<treated>.yaml arms/<null>.yaml \
  --repo ~/dev/haiku.rag --env ~/dev/haiku.rag/.env --detach <session>
```

Queue the whole night before the first arm starts. The queue chains on
completion, so no arm waits for a person to notice the previous one finished.
tmux is the process supervisor and nothing more: `tmux attach -t <session>`
to watch, detach with `C-b d`. Run it from the checkout that has the
harness, never from `/tmp`.

A fresh arm's `worktree` does not exist yet: the queue provisions it from
`--repo` at the pinned sha, then preflights it. So run `evaluations preflight`
by hand only against a checkout that already exists, and queue a fresh arm
straight away rather than preflighting it first, where the worktree check can
only fail.

For each arm the queue: preflights, runs the smoke and requires it to exit
cleanly with cases, registers the launch row, runs the arm from its worktree with output in
`~/.local/share/haiku.rag/evaluations/logs/<name>.log`, kills it at
`deadline_hours`, and completes the registry row from the trace. A missing
worktree is provisioned at the pinned sha with `uv sync` and the `.env`
copied. A failed step ends that arm; the queue moves on. An arm that fails
after it was registered keeps its launched row: complete it by hand with
`evaluations arms complete <name>`.

Provisioning by hand, when you must: `git worktree add --detach <path> <sha>`,
`uv sync` inside it, copy `.env` from a sibling. The third step is the one
that gets forgotten.

## 5. Watching a run

- **Progress is the run's result file, and `case:` spans in Logfire.** The
  file gains a row per finished case at
  `<data dir>/evaluations/results/<name>.partial.jsonl`, so `wc -l` on it is
  the progress count and it works with telemetry off. The progress bar renders
  only to a terminal; a redirected log and a tmux pane are empty until the run
  ends. The `debug-evals` skill has the span query.
- **The queue calls out a run that finishes no case for ten minutes** and does
  not touch it. Read the queue log for that line before deciding a run is
  healthy, and section 6 when it appears.
- **A rate needs a baseline of thirty minutes or the whole run so far**, and
  the report states the baseline with the rate. Two readings a few minutes
  apart catch a burst or a flat stretch and are wrong either way. Builds print
  their own cumulative line; read it instead of sampling.
- **A smoke's timing predicts nothing** about the full run.
- **Concurrency changes wall time.** Arms sharing an endpoint cost more per
  case than one arm alone. The registry records how many `evaluations run`
  processes were live at launch; quote it with any timing.
- **Wait on exit status, never on a pattern in stderr.** A connection error
  read as a result has happened. Branch on the command's status, and make a
  watcher report every terminal state: silence from one that greps for the
  happy path looks exactly like progress.
- **Snapshot the endpoints before and after**:
  `docker inspect <container> --format '{{.RestartCount}} {{.State.StartedAt}}'`.
  A restart voids the pair, not just the arm: both sides rerun on the
  restarted engine.

## 6. When a run looks stuck

Use instruments that cannot lie.

- `ps -o %cpu` is a lifetime average and reads near zero on a spin. Take the
  delta of `/proc/<pid>/stat` fields 14 and 15 over ten seconds, and the delta
  of `write_bytes` in `/proc/<pid>/io`. Work that progresses writes; a
  deadlock burns CPU and writes nothing.
- Thread stacks without root: register `faulthandler` on `SIGUSR1` in the
  harness, send it twice twenty seconds apart. Identical frames are a
  deadlock, moving frames are slow progress.
- `pgrep -f <pattern>` matches the shell that runs it and the `uv run`
  wrapper. Match the venv python or use the PID from the queue log.
- **Never kill by pattern on a shared box.** PID from the queue log, or the
  exact tmux session, only.
- An exact, repeatable failure count names its own cause. Three stalls at the
  same document count is that document, not a concurrency bug.
- A library that caches a broken worker poisons every later call in the
  process. Recovery is a new process: record the item, exit non-zero, let the
  supervisor restart and resume.

## 7. After a run

```bash
evaluations arms show <name>
evaluations arms pair <treated> <baseline>
evaluations arms export registry/arms.jsonl
```

Commit the export and the arm files in the data repository. The registry
database itself never enters git.

If the queue voided an arm, the row says why: `no telemetry`, `exit code N`,
`deadline of N h exceeded`. A void arm is never paired.

## 8. Reporting

- Every number states its corpus: database path, document and chunk counts,
  stored embedder. Corpora have eras; numbers across a rebuild do not compare.
- Report the judged rate and the floor together. Unjudged cases inflate the
  judged rate silently.
- A single arm is never a gate result. Only a pair with a null pair is.
- A ratio takes numerator and denominator from one cohort, and the report
  names the cohort.
- A claim about a population is measured on the population. Three hand checks
  size an investigation, not an effect.
- Before asserting a mechanism, read the code path end to end and name one
  observation that would disprove the claim. When evidence points one way,
  measure the other direction before withdrawing a concern.
- Every mechanism claim is verified by a second session, against the code
  path or the population, before it reaches Yiorgis.
- Retract in the durable record, not only in conversation.

## Amending this skill

This skill is expected to change as operators learn. A new trap or step is
written into this file in the session that learns it. Facts about what exists
go to `reference.md` in the data repository or to the registry, never here. Every amendment is reviewed
by a second session or by Yiorgis before it stands, as a commit to this
repository, and is dated below. An amendment that adds something an operator
must know also adds it to the acceptance test's list of things a fresh
session must not need to be told. After a procedural amendment the acceptance
test runs again on a fresh session.

`evaluations/tests/test_skills.py` refuses a skill file that carries a commit
sha, a trace id or a run date above this section.

## Change log

- 2026-09-17: created from the operator plan, sections 1.6, 2, 3.1 and 3.3,
  and the preflight notes.
