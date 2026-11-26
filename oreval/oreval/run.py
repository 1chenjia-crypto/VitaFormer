"""Command line: python -m oreval.run --config configs/<file>.json

Config keys (JSON):
  runs:           {run_name: path to readouts CSV}  (one per model x seed; each needs `image` + readout columns)
  metadata:       path to metadata CSV (one row per image; column `image`)
  readout:        readout column to evaluate (e.g. foreground_area_px)
  assay:          raw assay column (e.g. ATP)               | assay_is_normalised: true to skip assay normalisation
  control:        {"column": ..., "values": [...]}          rows whose column is in values are controls (metadata only!)
  normalise:      {"group_cols": [...], "mode": "ratio"|"difference", "undefined": "nan"|<number>}
  rows_filter:    optional {"column": ..., "values": [...]} rows evaluated (e.g. split == test)
  association:    {"cluster_cols": [...], "strata_col": ... or null, "nboot": 2000, "seed": ...}
  dose_direction: optional {"series_cols": [...], "conc_col": ..., "min_conc": 4, "min_abs_rho": 0.5,
                            "rows_column": ..., "rows_values": [...]}
  ranking:        optional {"unit_col": ..., "group_col": ..., "conc_col": ..., "rows_column": ..., "rows_values": [...],
                            "min_units": 5, "nboot": 2000, "seed": ...}
  output:         output folder (created; existing files are not overwritten)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import core


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True)
    cfg = json.loads(Path(ap.parse_args(argv).config).read_text())
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    meta = pd.read_csv(cfg['metadata'], low_memory=False)
    ctrl = cfg['control']; meta['is_control'] = meta[ctrl['column']].isin(ctrl['values'])
    nrm = cfg['normalise']; r = cfg['readout']
    if not cfg.get('assay_is_normalised', False):
        meta = core.control_normalise(meta, [cfg['assay']], nrm['group_cols'], mode=nrm['mode'], zero=nrm.get('undefined', 'nan'))
        assay = cfg['assay'] + '_rel'
    else:
        assay = cfg['assay']
    prepared, assoc, dose, rank, log = {}, [], [], [], {'config': cfg, 'undefined_normalisations': {}}
    for name, path in cfg['runs'].items():
        d = pd.read_csv(path, low_memory=False)[['image', r]].merge(meta, on='image', how='inner', validate='one_to_one')
        d = core.control_normalise(d, [r], nrm['group_cols'], mode=nrm['mode'], zero=nrm.get('undefined', 'nan'))
        log['undefined_normalisations'][name] = d.attrs['undefined']
        x = r + '_rel'
        if 'rows_filter' in cfg:
            f = cfg['rows_filter']; ev = d[d[f['column']].isin(f['values'])].copy()
        else:
            ev = d
        prepared[name] = ev
        a = cfg['association']
        t = core.association(ev, x, assay, a['cluster_cols'], a.get('nboot', 2000), a.get('seed', 20261109), a.get('strata_col'))
        assoc.append(t.assign(run=name))
        if 'dose_direction' in cfg:
            dd = cfg['dose_direction']
            de = ev[ev[dd['rows_column']].isin(dd['rows_values'])] if 'rows_column' in dd else ev
            _, s = core.dose_direction(de, dd['series_cols'], dd['conc_col'], x, assay, dd.get('min_conc', 4), dd.get('min_abs_rho', .5),
                                       dd.get('aggregate', 'median'))
            dose.append({'run': name, **s})
        if 'ranking' in cfg:
            k = cfg['ranking']
            rr = d[d[k['rows_column']].isin(k['rows_values'])] if 'rows_column' in k else d
            s, _ = core.within_group_rank(rr, k['unit_col'], k['group_col'], k['conc_col'], x, assay, k.get('min_units', 5),
                                          k.get('nboot', 2000), k.get('seed', 20261030))
            rank.append({'run': name, **s})
    pd.concat(assoc).to_csv(out / 'association.csv', index=False)
    a = cfg['association']
    rho, pairs = core.paired_association(prepared, 'image', r + '_rel', assay, a['cluster_cols'], a.get('nboot', 2000), a.get('seed', 20261109))
    pairs.to_csv(out / 'association_paired.csv', index=False)
    if dose:
        pd.DataFrame(dose).to_csv(out / 'dose_direction.csv', index=False)
    if rank:
        pd.DataFrame(rank).to_csv(out / 'within_group_ranking.csv', index=False)
        k = cfg['ranking']; rr = d[d[k['rows_column']].isin(k['rows_values'])] if 'rows_column' in k else d
        g, _ = core.general_baseline(rr, k['unit_col'], k['group_col'], k['conc_col'], assay, k.get('min_units', 5))
        pd.DataFrame([g]).to_csv(out / 'general_sensitivity_baseline.csv', index=False)
    groups = {}
    for run in rho.index:
        fam = cfg.get('families', {}).get(run, run.rsplit('_s', 1)[0])
        groups.setdefault(fam, []).append(rho[run])
    pd.DataFrame([{'family': f, **core.seed_range(v)} for f, v in groups.items()]).to_csv(out / 'seed_ranges_association.csv', index=False)
    (out / 'run_log.json').write_text(json.dumps(log, indent=2, default=str))
    print(pd.concat(assoc)[lambda t: t.stratum == 'pooled'].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
