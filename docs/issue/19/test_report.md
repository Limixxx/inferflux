# Issue #19 — S5 Overlap Scheduling Test Report

**Date:** 2026-09-02
**Branch:** issue-19
**Test Runner:** ts-node (native Node.js)

## TypeScript Compilation

| Check | Result |
|-------|--------|
| `tsc --noEmit` | PASS (0 errors) |
| `tsc` (build) | PASS (0 errors) |

## S5 Test Suite — sglang-s5.test.ts

**Total: 23 passed, 0 failed**

| # | Test Name | Result |
|---|-----------|--------|
| 1 | test_simulation_clock_advance | PASS |
| 2 | test_simulation_clock_schedule_gpu | PASS |
| 3 | test_simulation_clock_schedule_gpu_zero | PASS |
| 4 | test_simulation_clock_can_overlap | PASS |
| 5 | test_simulation_clock_on_tick_callback | PASS |
| 6 | test_overlap_short_prompt_normal | PASS |
| 7 | test_overlap_chunked_prefill_idle_flush | PASS |
| 8 | test_overlap_last_data_delay_graph_capture | PASS |
| 9 | test_overlap_last_data_delay_eager | PASS |
| 10 | test_overlap_empty_tick_flush_last_data | PASS |
| 11 | test_overlap_high_watermark_backpressure | PASS |
| 12 | test_overlap_high_watermark_zero_always_backpressure | PASS |
| 13 | test_overlap_finished_reqs_dedup | PASS |
| 14 | test_overlap_idle_count_for_flush_zero | PASS |
| 15 | test_overlap_empty_tick_no_last_data | PASS |
| 16 | test_simulation_clock_multiple_callbacks | PASS |
| 17 | test_simulation_clock_schedule_gpu_after_advance | PASS |
| 18 | test_clock_scheduler_integration | PASS |
| 19 | test_clock_auto_creation | PASS |
| 20 | test_clock_no_auto_creation_without_metrics | PASS |
| 21 | test_normal_tick_advances_clock | PASS |
| 22 | test_tick_counter_property | PASS |
| 23 | test_backpressure_process_last_data_still_works | PASS |

## Regression Tests

### S3 Test Suite — sglang-s3.test.ts
**Total: 52 passed, 0 failed**

### S4 Test Suite — sglang-s4.test.ts
**Total: 46 passed, 0 failed**

## Bugs Found & Fixed During Testing

### Bug 1: _lastOverlapData overwrite in overlap mode
- **Root cause:** `_overlapTick` scheduled new forward while GPU was busy (pending `_lastOverlapData` not yet processed), overwriting the previous result and causing data loss.
- **Fix:** Restructured `_overlapTick` Phase order — process pending results (Phase 2) before scheduling new forwards (Phase 3). Added `gpuBusy` guard (`_lastOverlapData !== null`) to prevent scheduling when GPU is occupied.

### Bug 2: finishedReqs set replaced instead of accumulated
- **Root cause:** `_processLastData` replaced `finishedReqs` with only newly-finished requests, losing previously-finished request tracking. Combined with overlap scheduling, finished requests could re-appear in subsequent batches and be double-freed.
- **Fix:** Added `req.finished` check in `_processLastData` to skip already-finished requests, preventing double resource release.

### Bug 3: Missing request submission in test 11
- **Root cause:** `test_overlap_finished_reqs_dedup` created `msgs` but never called `scheduler.runTick(msgs)` before `runUntilDone`.
- **Fix:** Added `scheduler.runTick(msgs)` call before `runUntilDone`.
