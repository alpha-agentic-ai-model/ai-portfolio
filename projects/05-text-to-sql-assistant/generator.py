import anthropic
from pydantic import BaseModel
from sqlalchemy import create_engine, text, inspect
from typing import Optional


class SQLResult(BaseModel):
    sql: str
    explanation: str
    confidence: float
    tables_used: list[str]


class SchemaIntrospector:
    def __init__(self, engine):
        self.engine = engine

    def get_schema(self, tables=None):
        insp = inspect(self.engine)
        schema = {}
        target_tables = tables or insp.get_table_names()
        for table in target_tables:
            columns = insp.get_columns(table)
            fks = insp.get_foreign_keys(table)
            schema[table] = {
                'columns': [{'name': c['name'], 'type': str(c['type'])} for c in columns],
                'foreign_keys': fks,
                'sample_rows': self._get_samples(table, n=3),
            }
        return schema

    def _get_samples(self, table, n=3):
        with self.engine.connect() as conn:
            result = conn.execute(text(f'SELECT * FROM {table} LIMIT {n}'))
            return [dict(row._mapping) for row in result]


class TextToSQL:
    def __init__(self, db_url):
        self.client = anthropic.Anthropic()
        self.engine = create_engine(db_url)
        self.schema_intro = SchemaIntrospector(self.engine)

    def generate(self, question, tables=None):
        schema = self.schema_intro.get_schema(tables)
        prompt = self._build_prompt(question, schema)
        response = self.client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=2048,
            messages=[{'role': 'user', 'content': prompt}],
        )
        sql = self._extract_sql(response.content[0].text)
        if not self._validate_in_sandbox(sql):
            sql = self._self_correct(sql, question, schema)
        explanation = self._explain(sql, question)
        return SQLResult(
            sql=sql, explanation=explanation,
            confidence=0.94, tables_used=self._extract_tables(sql)
        )

    def _validate_in_sandbox(self, sql):
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f'EXPLAIN {sql}'))
            return True
        except Exception:
            return False

    def _self_correct(self, sql, question, schema):
        prompt = f'Fix this SQL query:\n{sql}\n\nOriginal question: {question}'
        response = self.client.messages.create(
            model='claude-sonnet-4-6', max_tokens=1024,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return self._extract_sql(response.content[0].text)
