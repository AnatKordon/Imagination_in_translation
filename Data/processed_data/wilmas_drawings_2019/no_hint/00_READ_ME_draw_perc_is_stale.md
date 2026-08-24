# Heads up: `draw_perc/` in this arm is stale

As of 2026-08-24, `no_hint/draw_perc` holds 33 rows while `with_hint/draw_perc` holds 61.
The 28 drawings that became available when the perception source folder was replaced were
described into the with_hint arm only, by decision.

`draw_imm` (75) and `draw_del` (35) are complete and correct in this arm — nothing wrong
with them.

Details and how to fix: [`draw_perc/00_STALE_NOT_UPDATED.md`](draw_perc/00_STALE_NOT_UPDATED.md)
