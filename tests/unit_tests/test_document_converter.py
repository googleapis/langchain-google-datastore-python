# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mypy: disable-error-code="attr-defined"

import pytest
from google.cloud import datastore
from google.cloud.datastore.helpers import GeoPoint
from langchain_core.documents import Document

from langchain_google_datastore.document_converter import (
    DATASTORE_TYPE,
    GEOPOINT,
    KEY,
    convert_firestore_entity,
    convert_langchain_document,
)


@pytest.mark.parametrize(
    "entity_dict,langchain_doc",
    [
        (
            {
                "key": pytest.client.key(*("foo", "bar")),
                "properties": {"field_1": "data_1", "field_2": 2},
            },
            Document(
                page_content='{"field_1": "data_1", "field_2": 2}',
                metadata={
                    "key": {
                        "path": ("foo", "bar"),
                        DATASTORE_TYPE: KEY,
                    }
                },
            ),
        ),
        (
            {"key": pytest.client.key(*("foo", "bar")), "properties": {}},
            Document(
                page_content="{}",
                metadata={
                    "key": {
                        "path": ("foo", "bar"),
                        DATASTORE_TYPE: KEY,
                    }
                },
            ),
        ),
        (
            {
                "key": pytest.client.key(*("foo", "bar")),
                "properties": {
                    "field_1": GeoPoint(1, 2),
                    "field_2": [
                        "data",
                        2,
                        {"nested": pytest.client.key(*("abc", "xyz"))},
                    ],
                    "field_3": pytest.client.entity(
                        pytest.client.key("NestedKind", 123)
                    ),
                },
            },
            Document(
                page_content='{"field_1": {"latitude": 1, "longitude": 2, "datastore_type": "geopoint"}, '
                + '"field_2": ["data", 2, {"nested": {"key": ["abc", "xyz"], "datastore_type": "key"}}], '
                + '"field_3": {"key": ["NestedKind", 123], "properties": {}, "datastore_type": "entity"}}',
                metadata={
                    "key": {
                        "path": ("foo", "bar"),
                        DATASTORE_TYPE: KEY,
                    }
                },
            ),
        ),
    ],
)
def test_convert_firestore_entity_default_fields(entity_dict, langchain_doc) -> None:
    entity = pytest.client.entity(entity_dict["key"])
    entity.update(entity_dict["properties"])

    return_doc = convert_firestore_entity(entity)
    print("debug")
    print(return_doc)
    print(langchain_doc)

    assert return_doc == langchain_doc


@pytest.mark.parametrize(
    "entity_dict,langchain_doc,page_content_properties,metadata_properties",
    [
        (
            {
                "key": pytest.client.key(*("abc", "xyz")),
                "properties": {"data_field": "data", "extra_field": 1},
            },
            Document(
                page_content="data",
                metadata={
                    "key": {
                        "path": ("abc", "xyz"),
                        DATASTORE_TYPE: KEY,
                    },
                    "data_field": "data",
                },
            ),
            ["data_field"],
            ["data_field"],
        ),
        (
            {
                "key": pytest.client.key(*("abc", "xyz")),
                "properties": {"field_1": 1, "field_2": "val"},
            },
            Document(
                page_content="val",
                metadata={
                    "key": {
                        "path": ("abc", "xyz"),
                        DATASTORE_TYPE: KEY,
                    },
                    "field_1": 1,
                },
            ),
            ["field_2"],
            ["field_1"],
        ),
        (
            {
                "key": pytest.client.key(*("abc", "xyz")),
                "properties": {
                    "field_1": "val_1",
                    "field_2": "val_2",
                    "field_3": "val_3",
                    "field_4": "val_4",
                },
            },
            Document(
                page_content='{"field_2": "val_2", "field_3": "val_3"}',
                metadata={
                    "key": {
                        "path": ("abc", "xyz"),
                        DATASTORE_TYPE: KEY,
                    },
                    "field_1": "val_1",
                },
            ),
            ["field_2", "field_3"],
            ["field_1"],
        ),
        (
            {
                "key": pytest.client.key(*("abc", "xyz")),
                "properties": {
                    "field_1": "val_1",
                    "field_2": "val_2",
                    "field_3": "val_3",
                    "field_4": "val_4",
                },
            },
            Document(
                page_content='{"field_2": "val_2", "field_3": "val_3"}',
                metadata={
                    "key": {
                        "path": ("abc", "xyz"),
                        DATASTORE_TYPE: KEY,
                    },
                    "field_1": "val_1",
                    "field_4": "val_4",
                },
            ),
            [],
            ["field_1", "field_4"],
        ),
        (
            {
                "key": pytest.client.key(*("abc", "xyz")),
                "properties": {
                    "field_1": "val_1",
                    "field_2": "val_2",
                    "field_3": "val_3",
                    "field_4": "val_4",
                },
            },
            Document(
                page_content='{"field_2": "val_2", "field_4": "val_4"}',
                metadata={
                    "key": {
                        "path": ("abc", "xyz"),
                        DATASTORE_TYPE: KEY,
                    },
                    "field_1": "val_1",
                    "field_3": "val_3",
                },
            ),
            ["field_2", "field_4"],
            [],
        ),
    ],
)
def test_convert_firestore_entity_with_filters(
    entity_dict, langchain_doc, page_content_properties, metadata_properties
) -> None:
    entity = pytest.client.entity(entity_dict["key"])
    entity.update(entity_dict["properties"])

    return_doc = convert_firestore_entity(
        entity, page_content_properties, metadata_properties
    )

    assert return_doc == langchain_doc


@pytest.mark.parametrize(
    "langchain_doc,entity_dict",
    [
        (
            Document(
                page_content="value",
                metadata={
                    "key": {
                        "path": ("foo", "bar"),
                        DATASTORE_TYPE: "key",
                    }
                },
            ),
            {
                "key": {"path": ("foo", "bar"), DATASTORE_TYPE: KEY},
                "properties": {"page_content": "value"},
            },
        ),
        (
            Document(page_content="value", metadata={"key": {}}),
            {"key": None, "properties": {"page_content": "value", "key": {}}},
        ),
        (
            Document(
                page_content="value",
                metadata={"key": {"path": "foo/bar", "unexpected_field": "data"}},
            ),
            {
                "key": None,
                "properties": {
                    "page_content": "value",
                    "key": {"path": "foo/bar", "unexpected_field": "data"},
                },
            },
        ),
        (
            Document(
                page_content="value",
                metadata={
                    "key": {
                        "path": ("foo", "bar"),
                        DATASTORE_TYPE: KEY,
                    },
                    "metadata_field": {
                        "key": ("abc", "xyz"),
                        DATASTORE_TYPE: KEY,
                    },
                },
            ),
            {
                "key": {"path": ("foo", "bar"), DATASTORE_TYPE: KEY},
                "properties": {
                    "page_content": "value",
                    "metadata_field": pytest.client.key(*("abc", "xyz")),
                },
            },
        ),
        (
            Document(
                page_content='{"field_1": "val_1", "field_2": "val_2"}',
                metadata={
                    "key": {
                        "path": ("foo", "bar"),
                        DATASTORE_TYPE: KEY,
                    },
                    "field_3": "val_3",
                },
            ),
            {
                "key": {"path": ("foo", "bar"), DATASTORE_TYPE: KEY},
                "properties": {
                    "field_1": "val_1",
                    "field_2": "val_2",
                    "field_3": "val_3",
                },
            },
        ),
        (
            Document(
                page_content="",
                metadata={
                    "key": {
                        "path": ("foo", "bar"),
                        DATASTORE_TYPE: KEY,
                    }
                },
            ),
            {
                "key": {"path": ("foo", "bar"), DATASTORE_TYPE: KEY},
                "properties": {},
            },
        ),
        (
            Document(
                page_content="",
                metadata={
                    "key": {
                        "path": ("foo", "bar"),
                        DATASTORE_TYPE: KEY,
                    },
                    "point": {
                        "latitude": 1,
                        "longitude": 2,
                        DATASTORE_TYPE: GEOPOINT,
                    },
                    "field_2": "val_2",
                },
            ),
            {
                "key": {"path": ("foo", "bar"), DATASTORE_TYPE: KEY},
                "properties": {"point": GeoPoint(1, 2), "field_2": "val_2"},
            },
        ),
        (Document(page_content="", metadata={}), {"key": None, "properties": {}}),
        (
            Document(
                page_content='{"array":[1, "data", {"k_1":"v_1", "k_point": {"latitude": 1, "longitude": 0, "datastore_type": "geopoint"}}], "f_2": 2}',
                metadata={},
            ),
            {
                "key": None,
                "properties": {
                    "array": [1, "data", {"k_1": "v_1", "k_point": GeoPoint(1, 0)}],
                    "f_2": 2,
                },
            },
        ),
    ],
)
def test_convert_langchain_document(langchain_doc, entity_dict):
    return_doc = convert_langchain_document(langchain_doc, pytest.client)

    assert return_doc == entity_dict


@pytest.mark.parametrize(
    "entity_dict",
    [
        {
            "key": pytest.client.key(*("foo", "bar")),
            "properties": {
                "field_1": GeoPoint(1, 2),
                "field_2": [
                    "data",
                    2,
                    {"nested": pytest.client.key(*("abc", "xyz"))},
                ],
            },
        },
    ],
)
def test_roundtrip_firestore(entity_dict):
    key = entity_dict["key"]
    key_expected = {"path": ("foo", "bar"), DATASTORE_TYPE: KEY}
    entity = pytest.client.entity(key)
    entity.update(entity_dict["properties"])

    langchain_doc = convert_firestore_entity(entity)
    roundtrip_doc = convert_langchain_document(langchain_doc, pytest.client)

    assert roundtrip_doc["properties"] == entity_dict["properties"]
    assert roundtrip_doc["key"] == key_expected
