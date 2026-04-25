# Temporal Pooling And Bursting

This note explains the Layer 1 bursting issue seen while investigating
temporal pooling in `chat_htm`, what appears to cause it, where to look in the
code, and how to test it.

## Short Summary

The symptom is: enabling temporal pooling can increase column bursting in Layer
1 instead of smoothly improving temporal stability.

The main reason is that a column only avoids bursting when a cell was both:

- predictive at `t-1`
- backed by an active sequence segment at `t-1`

Temporal pooling can interfere with that in two ways:

1. It updates the same proximal permanence tensor already used by spatial
   learning, so TP can make columns win inhibition before distal sequence
   structure is strong enough to support non-bursting predictions.
2. If TP persistence is too aggressive, it can still amplify bursting pressure.
   The current calculator now carries forward recent segment evidence together
   with persistence-based predictive state so it no longer creates the old
   "predictive but still bursts" mismatch by itself.

The current calculator also applies a local trust gate to TP proximal updates:

- TP proximal reinforcement only applies to columns that had recent
  segment-backed temporal support
- this keeps stronger TP settings from directly reinforcing unstable columns
  into bursty winners

## Implemented Fixes

Two local fixes are now in place.

1. **Persistence must stay segment-backed.**

   `TemporalPoolerCalculator::update_distal()` now receives mutable
   `active_segs_time` from `PredictCellsCalculator`. When persistence extends a
   cell's predictive state from timestep `t-1` to `t`, it also copies that same
   cell's active segment timestamp from `t-1` to `t`.

   This matters because `ActiveCellsCalculator` avoids bursting on timestep
   `t+1` only if, at timestep `t`, the cell was both:

   - predictive in `predict_cells_time`
   - has at least one segment whose `active_segs_time` entry is `t`

   Without copying the segment timestamp, persistence could create a predictive
   bit that still fails the next burst-gate check.

   This is not claiming the segment became newly active from current synapse
   input. It deliberately treats the segment that was active at `t-1` as still
   active for one persisted prediction step. In other words, persistence is
   saying: "use the previous segment as the reason this cell remains
   predictive." That makes the timestamp a carried-forward temporal-pooling
   state, not a fresh `PredictCellsCalculator` segment activation.

   This is safe because `active_segs_time` stores the **last timestep a segment
   is considered active**, not the first timestep it became active. Normal
   prediction updates that timestamp when current active cells drive the segment
   over threshold. TP persistence updates that timestamp when the same segment
   was active at `t-1` and is being carried forward for one persistence step.
   So the meaning becomes: "latest timestep this segment is active, either from
   fresh distal input or from TP persistence."

2. **TP proximal learning is trust-gated.**

   `TemporalPoolerCalculator::update_proximal()` now receives read-only
   `predict_cells_time` and `active_segs_time`. Before applying TP proximal
   reinforcement to a column, it checks that at least one cell in that column
   was predictive and had an active segment at `t-1`.

   This keeps high `temporal_pooling.spatial_permanence_inc` values from
   reinforcing columns that are only spatial winners and do not yet have local
   distal temporal support.

Both fixes use local per-cell/per-column state already available in the layer.
No global burst-rate rule or layer-wide normalization was added.

## Why Bursting Happens

For newly active columns, the burst/no-burst decision is made in:

- `htm_flow/src/sequence_pooler/active_cells/active_cells.cpp`

The key rule is:

- if no cell in the column was predictive at `t-1` and had an active segment at
  `t-1`, the column bursts

Important supporting code paths:

- `htm_flow/src/htm_layer.cpp`
  - step order is: overlap -> inhibition -> spatial learning -> active cells ->
    predict cells -> sequence learning -> temporal pooler
  - TP runs after normal sequence memory work and writes back into shared state
- `htm_flow/src/temporal_pooler/temporal_pooler.cpp`
  - `update_proximal()` changes `col_syn_perm_`
  - `update_proximal()` now requires recent segment-backed temporal support
    before it applies TP proximal reinforcement
  - `update_distal()` changes `distal_synapses_`
  - optional persistence now carries forward the same cell's active segment
    timestamp when it extends predictive state
- `htm_flow/src/sequence_pooler/predict_cells/predict_cells.cpp`
  - `active_segs_time_` is produced here from connected distal support
  - `get_active_segs_time_mutable()` exposes that timestamp buffer so TP
    persistence can carry a real segment timestamp forward

## Important Investigation Notes

- Temporal pooling is not a separate clean layer of logic. It shares learning
  tensors with the normal layer algorithm.
- The burst spike observed when turning TP on later was useful as a diagnosis,
  but the real goal is a static config that works from the start.
- At the moment, the safest configuration is still to keep TP enabled from the
  beginning, keep TP proximal learning weak, and let TP distal learning
  dominate early.
- Because TP proximal is now trust-gated, stronger TP settings can be explored
  with less risk of immediately reintroducing Layer 1 burst spikes.
- Persistence bookkeeping is now internally consistent with the burst gate, but
  persistence is still disabled by default because it is more fragile than base
  TP learning.
- Carrying a segment timestamp forward is intentionally narrow: it only happens
  for a cell that already had an active segment on the previous timestep. This
  makes persistence visible to the next burst-gate check, but it also means
  active-segment timestamps can now be refreshed by either normal prediction or
  TP persistence. Keep this in mind when debugging segment traces: the timestamp
  means "last considered active," not "freshly activated by current input."
- If someone wants to strengthen TP persistence further in the future, the
  first place to inspect is the persistence logic in
  `htm_flow/src/temporal_pooler/temporal_pooler.cpp`, because that is where TP
  predictive state is coupled to segment evidence.

## Current Config Direction

The current reference config is:

- `configs/word_rows_2layer_text.yaml`

Layer 1 temporal pooling is enabled from the start with:

- `enable_persistence: false`
- small TP proximal learning
- stronger TP distal learning

Generic `HTMLayerConfig` defaults now also bias toward safer TP startup:

- `temp_enable_persistence: false`
- `temp_sequence_permanence_inc > temp_spatial_permanence_inc`

This is meant to avoid the need for timestep-specific runtime patching while
still allowing temporal pooling to learn.

## How To Test

The main regression coverage is in:

- `tests/integration/test_text_htm.cpp`

Relevant tests:

- `TextHTMIntegration.RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting`
  - keeps the original diagnostic scenario where TP is applied at runtime
- `TextHTMIntegration.AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP`
  - compares always-on TP against a TP-off control
- `TextHTMIntegration.AlwaysOnTemporalPoolingWithPersistenceDoesNotRaiseLayer1BurstingVsNoTP`
  - verifies that TP persistence no longer reintroduces a large Layer 1 burst spike
- `TextHTMIntegration.StrongTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP`
  - verifies that a much stronger Layer 1 TP setup does not automatically bring
    back the old burst failure mode

Calculator-level coverage lives in:

- `htm_flow/test/unit/test_temporal_pooler.cpp`
  - checks that persistence now requires recent segment evidence
  - checks that persistence carries that segment evidence forward
  - checks the proximal trust gate and the burst guards for both Rule A and Rule B

Run them with:

```bash
cmake --build build --target chat_htm_tests
./build/chat_htm_tests --gtest_filter=TextHTMIntegration.RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting:TextHTMIntegration.AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP
```

For stronger TP tuning experiments, also run:

```bash
./build/chat_htm_tests --gtest_filter=TextHTMIntegration.AlwaysOnTemporalPoolingWithPersistenceDoesNotRaiseLayer1BurstingVsNoTP:TextHTMIntegration.StrongTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP
```

## What To Look At During Debugging

When investigating this issue again, check:

- whether Layer 1 active columns are increasing without a matching increase in
  predictive coverage
- whether bursting rises because TP proximal learning is bypassing the intended
  local temporal-support gate
- whether persistence was enabled and is producing predictive state without
  valid segment evidence
- whether distal learning rates and thresholds make it too hard for TP-created
  synapses to become useful before new columns start winning

## Future Work

Future improvements should focus on making temporal learning stronger without
reintroducing burst spikes.

The likely next step is not a larger config sweep, but a closer look at whether
TP proximal learning can be made even less coupled to the shared
spatial-pooling permanence tensor without losing the local-learning character
of the algorithm.
