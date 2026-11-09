"""Full deploy-protocol run (Phase 3/4): 300-config seeded grid, long-only,
single vs vote vs blend ensemble OOS, prints acceptance-gate checks."""
import time
import numpy as np
import config
from data import load_or_fetch
from grid import (prepare_dataset, run_grid, final_walkforward,
                  diverse_subset, combined_walkforward)
from search_space import build_configs


def summarize(tag, folds, rolled):
    sh = [m["sharpe"] for m in folds]
    med = float(np.median(sh)) if sh else float("nan")
    pos = sum(1 for x in sh if x > 0)
    print(f"[{tag}] pools={rolled['sharpe']:.2f} folds={[round(x,2) for x in sh]} "
          f"fold_med={med:.2f} pos={pos}/{len(sh)} trades={rolled.get('trades',0)}")


def main():
    t0 = time.time()
    config.configure_timeframe("4h", long_only=True)
    config.GRID.max_configs = 300
    df = load_or_fetch(days=365, force_refresh=False, timeframe=config.TIMEFRAME)
    X, arrs = prepare_dataset(df)
    configs = build_configs()[:300]
    print(f"grid over {len(configs)} configs, {X.shape}", flush=True)
    ranked = run_grid(X, df, arrs, configs=configs, n_jobs=12)
    for c, s in ranked[:10]:
        print(f"  rank {s:.3f}  {c['model']:<4} {c['label']['method']:<15} "
              f"h={c['label']['horizon']} m={c['margin']} gate={c.get('trend_gate')}", flush=True)
    c = ranked[0][0]
    f, fm, tb, rolled = final_walkforward(c, X, df, arrs)
    summarize("single", fm, rolled)
    members = diverse_subset([x[0] for x in ranked], 5)
    f, fm, tb, rolled = combined_walkforward(members, X, df, arrs, combine="vote")
    summarize("vote._ens", fm, rolled)
    f, fm, tb, rolled = combined_walkforward(members, X, df, arrs, combine="blend")
    summarize("blend_ens", fm, rolled)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()