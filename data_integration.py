"""Bounded tabular imports and explicit, validated data integration.

Archives are read in memory, never extracted. No uploaded code is executed.
"""
from __future__ import annotations
import io
import json
from pathlib import PurePosixPath
import zipfile
import pandas as pd

FORMATS = {'.csv', '.tsv', '.txt', '.xlsx', '.xls', '.json', '.jsonl', '.ndjson', '.parquet', '.feather'}
MAX_FILE = 25 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024
MAX_TABLES = 100
MAX_ROWS = 1_000_000
MAX_CELLS = 10_000_000


def validate(df):
    if df.empty or not len(df.columns):
        raise ValueError('Table is empty.')
    if len(df) > MAX_ROWS or df.size > MAX_CELLS:
        raise ValueError('Table exceeds the 1 million row / 10 million cell limit.')
    used, columns = set(), []
    for column in df.columns:
        base = str(column).strip() or 'Unnamed'
        name, n = base, 2
        while name in used:
            name = f'{base}_{n}'
            n += 1
        columns.append(name)
        used.add(name)
    df.columns = columns
    # Nested JSON values remain inspectable and hashable for profiling.
    for column in df.select_dtypes(include='object'):
        df[column] = df[column].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    return df


def read_tables(name, content):
    ext = PurePosixPath(name).suffix.lower()
    stream = io.BytesIO(content)
    if ext in {'.csv', '.tsv', '.txt'}:
        try:
            text = content.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = content.decode('cp1252')
        df = pd.read_csv(io.StringIO(text), sep='\t' if ext == '.tsv' else (',' if ext == '.csv' else None), engine='python')
    elif ext in {'.xls', '.xlsx'}:
        with pd.ExcelFile(stream) as book:
            if len(book.sheet_names) > MAX_TABLES:
                raise ValueError('Workbook has too many sheets.')
            return [(name if len(book.sheet_names) == 1 else f'{name}::{sheet}', validate(book.parse(sheet)))
                    for sheet in book.sheet_names]
    elif ext in {'.jsonl', '.ndjson'}:
        df = pd.read_json(stream, lines=True)
    elif ext == '.json':
        value = json.loads(content)
        if isinstance(value, list):
            df = pd.json_normalize(value)
        elif isinstance(value, dict):
            df = pd.DataFrame(value)
        else:
            raise ValueError('JSON must contain records or column arrays.')
    elif ext == '.parquet':
        df = pd.read_parquet(stream)
    elif ext == '.feather':
        df = pd.read_feather(stream)
    else:
        raise ValueError('Unsupported tabular format.')
    return [(name, validate(df))]


def import_files(files):
    """Return catalog and diagnostics; reject unsafe/oversized batches atomically."""
    catalog, ignored = {}, []
    total, members = 0, 0

    def add(name, content):
        nonlocal total, members
        members += 1
        total += len(content)
        if members > MAX_TABLES or total > MAX_TOTAL or len(content) > MAX_FILE:
            raise ValueError('Import exceeds 100 files, 25 MB per file, or 100 MB expanded total.')
        if PurePosixPath(name).suffix.lower() not in FORMATS:
            ignored.append(f'{name}: unsupported format')
            return
        try:
            tables = read_tables(name, content)
        except Exception as exc:
            ignored.append(f'{name}: {type(exc).__name__}: {exc}')
            return
        for label, frame in tables:
            candidate, n = label, 2
            while candidate in catalog:
                candidate = f'{label} ({n})'
                n += 1
            catalog[candidate] = frame
        if len(catalog) > MAX_TABLES or sum(df.size for df in catalog.values()) > MAX_CELLS:
            raise ValueError('Import exceeds 100 tables or 10 million cells.')

    for name, content in files:
        if len(content) > MAX_FILE:
            raise ValueError('Each upload must be at most 25 MB.')
        name = name.replace('\\', '/')
        if PurePosixPath(name).suffix.lower() != '.zip':
            add(name, content)
            continue
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if len(archive.infolist()) > MAX_TABLES:
                raise ValueError('ZIP contains more than 100 entries.')
            expanded = sum(info.file_size for info in archive.infolist())
            if total + expanded > MAX_TOTAL:
                raise ValueError('ZIP exceeds the 100 MB expanded import limit.')
            for info in archive.infolist():
                path = PurePosixPath(info.filename.replace('\\', '/'))
                if path.is_absolute() or '..' in path.parts or ':' in str(path) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('ZIP contains an unsafe path or symbolic link.')
                if info.is_dir():
                    continue
                if info.file_size > MAX_FILE or info.flag_bits & 1:
                    raise ValueError('ZIP contains an oversized or encrypted member.')
                if path.suffix.lower() == '.zip':
                    ignored.append(f'{path}: nested archives are not supported')
                    continue
                add(f'{name}/{path}', archive.read(info))
    if not catalog:
        raise ValueError('No readable tabular datasets found. ' + '; '.join(ignored[:3]))
    return catalog, ignored


def combine_matching(catalog):
    """Append only identical column sets, retaining every original table."""
    groups = {}
    for name, frame in catalog.items():
        groups.setdefault(frozenset(frame.columns), []).append((name, frame))
    added = []
    for group in groups.values():
        if len(group) < 2:
            continue
        source = '_source_file'
        while source in group[0][1].columns:
            source = '_' + source
        combined = validate(pd.concat([frame.assign(**{source: name}) for name, frame in group], ignore_index=True))
        label = 'Combined: ' + ', '.join(name for name, _ in group)
        while label in catalog:
            label += ' (combined)'
        catalog[label] = combined
        added.append(label)
    return added


def join_tables(catalog, left, right, keys, how='left'):
    if left not in catalog or right not in catalog or left == right:
        raise ValueError('Choose two different imported tables by their exact names.')
    if how not in {'left', 'inner', 'outer'} or not keys:
        raise ValueError('Choose left, inner or outer join and at least one key.')
    a, b = catalog[left], catalog[right]
    if any(key not in a or key not in b for key in keys):
        raise ValueError('Every join key must exist in both tables.')
    if a[keys].isna().any().any() or b[keys].isna().any().any():
        raise ValueError('Join keys contain missing values. Clean them before joining.')
    if a.duplicated(keys).any() and b.duplicated(keys).any():
        raise ValueError('Many-to-many joins are blocked to prevent inflated counts. Use unique keys.')
    if len(a) + len(b) > MAX_ROWS:
        raise ValueError('Join exceeds the row budget.')
    result = a.merge(b, on=keys, how=how, suffixes=('_left', '_right'),
                     validate='one_to_many' if not a.duplicated(keys).any() else 'many_to_one')
    return validate(result)
