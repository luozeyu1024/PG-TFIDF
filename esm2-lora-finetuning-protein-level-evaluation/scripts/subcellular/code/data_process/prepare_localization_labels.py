"""Apply the original localization matcher to an explicitly selected table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from Sub_cc_match_subcellular import add_label_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence_table', type=Path, required=True)
    parser.add_argument('--label_vocabulary', type=Path, required=True)
    parser.add_argument('--species', required=True)
    parser.add_argument('--output_table', type=Path, required=True)
    parser.add_argument('--counts_json', type=Path, required=True)
    args = parser.parse_args()
    targets = [args.output_table.resolve(), args.counts_json.resolve()]
    inputs = [args.sequence_table.resolve(), args.label_vocabulary.resolve()]
    if len(set(targets)) != 2 or any(p in inputs for p in targets):
        parser.error('Inputs and output paths must be distinct.')
    if any(p.exists() for p in targets):
        parser.error('Output already exists; choose a new output path.')
    vocabulary = json.loads(args.label_vocabulary.read_text(encoding='utf-8-sig'))
    labels = vocabulary[args.species]
    table = pd.read_excel(args.sequence_table)
    required = {'Entry', 'Sequence', 'Subcellular location [CC]'}
    if not required.issubset(table.columns):
        parser.error('Required columns: Entry, Sequence, Subcellular location [CC].')
    if not table['Entry'].is_unique or table['Entry'].isna().any():
        parser.error('Entry identifiers must be unique and nonmissing.')
    if any(label in table.columns for label in labels):
        parser.error('Input already contains localization-label columns.')
    result = add_label_columns(table.copy(), labels, 'Subcellular location [CC]')
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(args.output_table, index=False)
    summary = {
        'species': args.species,
        'rows': len(result),
        'label_names': labels,
        'positive_counts': {label: int(result[label].sum()) for label in labels},
        'zero_label_rows': int((result[labels].sum(axis=1) == 0).sum()),
    }
    args.counts_json.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
