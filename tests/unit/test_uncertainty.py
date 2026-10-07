"""
Unit tests for Uncertainty Decomposition and Targeted Clarification.
"""

import pytest
from src.agent.uncertainty import (
    UncertaintyDecomposer,
    UncertaintyDecomposition,
    DimensionUncertainty,
)


@pytest.fixture
def sample_schema():
    return {
        "Customer": ["CustomerId", "FirstName", "LastName", "Company", "Country"],
        "Invoice": ["InvoiceId", "CustomerId", "InvoiceDate", "Total"],
        "InvoiceLine": ["InvoiceLineId", "InvoiceId", "TrackId", "UnitPrice", "Quantity"],
        "Track": ["TrackId", "Name", "AlbumId", "MediaTypeId", "GenreId", "UnitPrice"],
    }


@pytest.fixture
def sample_fks():
    return [
        {"from_table": "Invoice", "to_table": "Customer", "from_col": "CustomerId", "to_col": "CustomerId"},
        {"from_table": "InvoiceLine", "to_table": "Invoice", "from_col": "InvoiceId", "to_col": "InvoiceId"},
        {"from_table": "InvoiceLine", "to_table": "Track", "from_col": "TrackId", "to_col": "TrackId"},
    ]


def test_empty_or_none_sql(sample_schema):
    decomposer = UncertaintyDecomposer()
    res = decomposer.decompose(
        question="Show me tracks",
        sql=None,
        schema_tables=sample_schema,
    )
    assert res.composite_uncertainty == 1.0
    assert res.composite_confidence == 0.0
    assert res.needs_clarification is True
    assert res.clarification is not None
    assert len(res.clarification.options) > 0


def test_high_certainty_single_table(sample_schema):
    decomposer = UncertaintyDecomposer()
    sql = "SELECT CustomerId, FirstName, LastName FROM Customer WHERE Country = 'USA'"
    res = decomposer.decompose(
        question="List all customers from USA",
        sql=sql,
        schema_tables=sample_schema,
    )
    assert res.composite_uncertainty < 0.25
    assert res.composite_confidence > 0.75
    assert res.needs_clarification is False
    assert res.schema_linking.uncertainty == 0.0
    assert res.join_path.uncertainty == 0.0


def test_cartesian_join_triggers_join_uncertainty(sample_schema):
    decomposer = UncertaintyDecomposer()
    sql = "SELECT Customer.FirstName, Invoice.Total FROM Customer, Invoice"
    res = decomposer.decompose(
        question="Show customers and invoices",
        sql=sql,
        schema_tables=sample_schema,
    )
    assert res.join_path.uncertainty >= 0.8
    assert "Cartesian join" in res.join_path.reasons[0]
    assert res.dominant_dimension == "join_path"
    assert res.needs_clarification is True


def test_missing_aggregation_triggers_agg_uncertainty(sample_schema):
    decomposer = UncertaintyDecomposer()
    # Question asks for total sales, but SQL just selects rows without SUM or GROUP BY
    sql = "SELECT Total FROM Invoice"
    res = decomposer.decompose(
        question="What is the total sales amount?",
        sql=sql,
        schema_tables=sample_schema,
    )
    assert res.aggregation.uncertainty >= 0.4
    assert res.dominant_dimension == "aggregation"


def test_ungrounded_filter_value_triggers_value_uncertainty(sample_schema):
    decomposer = UncertaintyDecomposer()
    # Query filters on 'Platinum' status which was never mentioned in the prompt
    sql = "SELECT CustomerId FROM Customer WHERE Company = 'Platinum'"
    res = decomposer.decompose(
        question="Show me top VIP accounts",
        sql=sql,
        schema_tables=sample_schema,
    )
    assert res.value_grounding.uncertainty >= 0.25
    assert any("Platinum" in r for r in res.value_grounding.reasons)


def test_to_dict_serialization(sample_schema):
    decomposer = UncertaintyDecomposer()
    sql = "SELECT CustomerId FROM Customer"
    res = decomposer.decompose(question="Show customers", sql=sql, schema_tables=sample_schema)
    d = res.to_dict()
    assert "composite_uncertainty" in d
    assert "composite_confidence" in d
    assert "dimensions" in d
    assert "schema_linking" in d["dimensions"]
