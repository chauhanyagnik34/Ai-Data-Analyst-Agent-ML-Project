# v4 multi-dataset imports

Install `requirements.txt` and `requirements-v4.txt`, then start the existing API and frontend as usual.

The existing Browse files, Browse folder, and drag/drop controls accept CSV, TSV, delimited TXT, Excel (.xls/.xlsx), JSON records or column arrays, JSON Lines (.jsonl/.ndjson), Parquet, Feather, and ZIP archives containing these formats. Every nonempty Excel sheet is imported. JSON record objects are flattened; remaining nested values are represented as JSON strings. PDFs, images, SQL dumps, executable formats, and nested archives are not supported. Import diagnostics identify skipped files.

Single tables work as before. Tables with identical column names are appended automatically, with `_source_file` provenance (prefixed again if that name exists). Originals remain available in Active Dataset. This is schema matching, not semantic matching: only upload matching tables together when they represent the same kind of record. No rows are deduplicated automatically. When there are multiple schema groups, the first combined group becomes active; all others remain selectable. Unrelated tables are never joined automatically.

To combine related tables, type an explicit command into the existing chat:

```
join "orders.csv" with "customers.csv" on customer_id
inner join "orders.csv" with "customers.csv" on customer_id
outer join "orders.csv" with "customers.csv" on customer_id, region
```

Use the exact names in Active Dataset (including ZIP/folder prefixes and sheet names). Left join is the default. One side must have unique keys; missing keys and many-to-many joins are rejected. Shared non-key columns receive `_left` / `_right` suffixes. Keys must have compatible types. The joined result becomes active for subsequent normal analysis, charts, transformations, and CSV export. Originals remain available in the existing selector.

The same operation is available through `POST /api/integrate` with `session_id`, `left`, `right`, `keys` (array), and optional `how` (`left`, `inner`, `outer`). This does not require Ollama; ordinary natural-language analysis still uses the existing planner.

Limits: 100 uploaded files, 25 MB per upload/archive member, 100 MB total expanded input, 100 imported tables, 1 million rows per table, and 10 million imported cells. Archives are never extracted and unsafe paths, symlinks and encrypted members are rejected. Parsed table limits are checked after decoding; compressed columnar data may require more memory than its file size. Failed batches leave the prior dataset intact. Valid files can load alongside unsupported/malformed files, which are reported.

Recent Chats now has a delete button per row, with confirmation. Deleting the active chat starts an empty analysis. Other chats are retained; stale entries can also be deleted after a server restart. Sessions remain in server memory as in v4 and are not durable across restarts.

Validation: `python -m pytest tests/test_data_integration.py` and `cd frontend && npm run build`.
