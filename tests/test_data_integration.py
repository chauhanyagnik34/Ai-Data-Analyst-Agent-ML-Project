import io
import zipfile
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from data_integration import import_files, combine_matching, join_tables
from insightforge_api import app, _sessions


def archive(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        for name, data in entries:
            z.writestr(name, data)
    return buffer.getvalue()


def test_formats_and_provenance():
    catalog, ignored = import_files([('a.csv', b'id,value\n1,10\n'), ('folder/b.tsv', b'id\tvalue\n2\t20\n')])
    combined = combine_matching(catalog)
    assert len(catalog) == 3
    assert catalog[combined[0]]['value'].sum() == 30
    assert catalog[combined[0]]['_source_file'].tolist() == ['a.csv', 'folder/b.tsv']
    assert not ignored


def test_zip_json_and_duplicate_names():
    catalog, ignored = import_files([('data.zip', archive([
        ('a/records.json', '[{"id":1,"profile":{"city":"A"}}]'),
        ('b/records.jsonl', '{"id":2,"value":4}\n'), ('readme.md', 'notes')]))])
    assert len(catalog) == 2
    assert 'profile.city' in catalog['data.zip/a/records.json']
    assert len(ignored) == 1


@pytest.mark.parametrize('name', ['../escape.csv', '/absolute.csv', 'C:/evil.csv'])
def test_zip_paths_rejected(name):
    with pytest.raises(ValueError, match='unsafe'):
        import_files([('bad.zip', archive([(name, 'a\n1')]))])


def test_excel_all_sheets_and_columnar_formats():
    b = io.BytesIO()
    with pd.ExcelWriter(b, engine='openpyxl') as writer:
        pd.DataFrame({'id': [1]}).to_excel(writer, sheet_name='one', index=False)
        pd.DataFrame({'name': ['A']}).to_excel(writer, sheet_name='two', index=False)
    catalog, _ = import_files([('book.xlsx', b.getvalue())])
    assert set(catalog) == {'book.xlsx::one', 'book.xlsx::two'}
    for ext in ('parquet', 'feather'):
        stream = io.BytesIO()
        getattr(pd.DataFrame({'id': [1]}), 'to_' + ext)(stream)
        tables, _ = import_files([('data.' + ext, stream.getvalue())])
        assert tables['data.' + ext]['id'].tolist() == [1]


def test_join_and_many_to_many_protection():
    catalog = {'orders.csv': pd.DataFrame({'customer_id': [1, 1, 2], 'amount': [10, 20, 30]}),
               'customers.csv': pd.DataFrame({'customer_id': [1, 2], 'name': ['A', 'B']})}
    result = join_tables(catalog, 'orders.csv', 'customers.csv', ['customer_id'])
    assert result.groupby('name')['amount'].sum().to_dict() == {'A': 30, 'B': 30}
    catalog['customers.csv'] = pd.concat([catalog['customers.csv']] * 2)
    with pytest.raises(ValueError, match='Many-to-many'):
        join_tables(catalog, 'orders.csv', 'customers.csv', ['customer_id'])


def test_api_upload_join_export_delete_and_single_regression():
    client = TestClient(app)
    sid = client.post('/api/session').json()['session_id']
    result = client.post('/api/upload', data={'session_id': sid}, files=[
        ('files', ('orders.csv', b'customer_id,amount\n1,10\n1,20\n2,30')),
        ('files', ('customers.csv', b'customer_id,name\n1,A\n2,B'))])
    assert result.status_code == 200, result.text
    assert len(result.json()['imported']) == 2
    result = client.post('/api/query', json={'session_id': sid, 'question': 'join "orders.csv" with "customers.csv" on customer_id'})
    assert result.status_code == 200, result.text
    assert result.json()['dataset']['rows'] == 3
    assert 'name' in result.json()['table']['columns']
    assert len(result.json()['imported']) == 3
    assert 'amount' in client.get('/api/export/' + sid).text
    assert client.post('/api/select-dataset', json={'session_id': sid, 'name': 'customers.csv'}).status_code == 200
    single = client.post('/api/upload', data={'session_id': sid}, files={'files': ('data.csv', b'id,value\n1,10')})
    assert single.json()['dataset']['columns'] == 2
    assert single.json()['imported'] == ['data.csv']
    assert client.delete('/api/session/' + sid).status_code == 200
    assert sid not in _sessions
    assert client.get('/api/session/' + sid).status_code == 404
    assert client.delete('/api/session/' + sid).status_code == 200


def test_failed_upload_preserves_session():
    client = TestClient(app)
    sid = client.post('/api/session').json()['session_id']
    client.post('/api/upload', data={'session_id': sid}, files={'files': ('data.csv', b'id\n1')})
    result = client.post('/api/upload', data={'session_id': sid}, files={'files': ('bad.zip', b'broken')})
    assert result.status_code == 400
    assert client.get('/api/session/' + sid).json()['dataset']['rows'] == 1
    assert client.get('/api/session/' + sid).json()['dataset']['preview']['columns'] == ['id']
    client.delete('/api/session/' + sid)
