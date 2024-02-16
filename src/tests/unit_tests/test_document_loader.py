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

import time
import unittest.mock as mock
from unittest import TestCase

import pytest
from google.cloud.datastore import Client
from langchain_core.documents import Document

from langchain_google_datastore import DatastoreLoader, DatastoreSaver


@pytest.fixture
def test_case() -> TestCase:
    return TestCase()


def test_firestore_write_roundtrip_and_load() -> None:
    saver = DatastoreSaver("WriteRoundTrip")
    loader = DatastoreLoader("WriteRoundTrip")

    docs = [Document(page_content="data", metadata={})]

    saver.upsert_documents(docs)
    # wait 1s for consistency
    time.sleep(1)
    written_docs = loader.load()
    saver.delete_documents(written_docs)
    # wait 1s for consistency
    time.sleep(1)

    deleted_docs = loader.load()

    assert len(written_docs) == 1
    assert written_docs[0].page_content == "data"
    assert written_docs[0].metadata != {}
    assert "key" in written_docs[0].metadata
    assert len(deleted_docs) == 0


def test_firestore_write_load_batch(test_case: TestCase) -> None:
    saver = DatastoreSaver("WriteBatch")
    loader = DatastoreLoader("WriteBatch")
    NUM_DOCS = 1000

    docs = []
    expected_docs = []
    for i in range(NUM_DOCS):
        docs.append(Document(page_content=f"content {i}"))
        expected_docs.append(
            Document(page_content=f"content {i}", metadata={"key": mock.ANY})
        )

    saver.upsert_documents(docs)
    # wait 5s for consistency
    time.sleep(5)
    docs_after_write = loader.load()
    saver.delete_documents(docs_after_write)
    # wait 5s for consistency
    time.sleep(5)
    docs_after_delete = loader.load()

    test_case.assertCountEqual(expected_docs, docs_after_write)
    assert len(docs_after_delete) == 0


def test_firestore_write_with_key(test_case: TestCase) -> None:
    saver = DatastoreSaver()
    loader = DatastoreLoader("WriteRef")

    expected_doc = [
        Document(
            page_content='{"f1": 1, "f2": 2}',
            metadata={"key": {"path": ("WriteRef", "doc"), "datastore_type": "key"}},
        )
    ]
    saver.upsert_documents(expected_doc)
    # wait 1s for consistency
    time.sleep(1)
    written_doc = loader.load()
    saver.delete_documents(written_doc)
    # wait 1s for consistency
    time.sleep(1)
    deleted_doc = loader.load()

    test_case.assertCountEqual(expected_doc, written_doc)
    assert len(deleted_doc) == 0


def test_firestore_delete_with_keys(test_case: TestCase) -> None:
    saver = DatastoreSaver()
    loader = DatastoreLoader("WriteKind")

    doc_to_insert = [
        Document(
            page_content='{"f1": 1, "f2": 2}',
            metadata={"key": {"path": ("WriteKind", "bar"), "datastore_type": "key"}},
        )
    ]

    expected_doc = [
        Document(
            page_content='{"f1": 1, "f2": 2}',
            metadata={"key": {"path": ("WriteKind", "bar"), "datastore_type": "key"}},
        )
    ]
    keys = [["WriteKind", "bar"]]
    saver.upsert_documents(documents=doc_to_insert)
    # wait 1s for consistency
    time.sleep(1)
    written_doc = loader.load()
    saver.delete_documents([], keys)
    # wait 1s for consistency
    time.sleep(1)
    deleted_doc = loader.load()

    test_case.assertCountEqual(expected_doc, written_doc)
    assert len(deleted_doc) == 0


@pytest.mark.parametrize(
    "page_properties,metadata_properties,expected_page_content,expected_metadata",
    [
        ([], [], '{"f1": "v1", "f2": "v2", "f3": "v3"}', {"key": mock.ANY}),
        (["f1"], [], "v1", {"key": mock.ANY, "f2": "v2", "f3": "v3"}),
        ([], ["f2"], '{"f1": "v1", "f3": "v3"}', {"key": mock.ANY, "f2": "v2"}),
        (["f1"], ["f2"], "v1", {"key": mock.ANY, "f2": "v2"}),
        (["f2"], ["f2"], "v2", {"key": mock.ANY, "f2": "v2"}),
    ],
)
def test_firestore_load_with_fields(
    page_properties,
    metadata_properties,
    expected_page_content,
    expected_metadata,
    test_case,
):
    saver = DatastoreSaver("WritePageFields")
    loader = DatastoreLoader(
        source="WritePageFields",
        page_content_properties=page_properties,
        metadata_properties=metadata_properties,
    )

    doc_to_insert = [
        Document(page_content='{"f1": "v1", "f2": "v2", "f3": "v3"}', metadata={})
    ]
    expected_doc = [
        Document(page_content=expected_page_content, metadata=expected_metadata)
    ]

    saver.upsert_documents(doc_to_insert)
    # wait 1s for consistency
    time.sleep(1)
    loaded_doc = loader.load()
    saver.delete_documents(loaded_doc)
    # wait 1s for consistency
    time.sleep(1)
    deleted_docs = loader.load()

    test_case.assertCountEqual(expected_doc, loaded_doc)
    assert len(deleted_docs) == 0


def test_firestore_load_from_query(test_case: TestCase):
    saver = DatastoreSaver("WriteQuery")
    loader_cleanup = DatastoreLoader("WriteQuery")

    docs_to_insert = [
        Document(page_content='{"num": 20, "region": "west_coast"}'),
        Document(page_content='{"num": 20, "region": "south_coast"}'),
        Document(page_content='{"num": 30, "region": "west_coast"}'),
        Document(page_content='{"num": 0, "region": "east_coast"}'),
    ]
    expected_docs = [
        Document(
            page_content='{"num": 20, "region": "west_coast"}',
            metadata={"key": mock.ANY},
        ),
        Document(
            page_content='{"num": 30, "region": "west_coast"}',
            metadata={"key": mock.ANY},
        ),
    ]
    query_load = pytest.client.query(kind="WriteQuery")
    query_load.add_filter("region", "=", "west_coast")
    loader = DatastoreLoader(query_load)

    saver.upsert_documents(docs_to_insert)
    # wait 1s for consistency
    time.sleep(1)
    loaded_docs = loader.load()
    saver.delete_documents(loader_cleanup.load())
    # wait 1s for consistency
    time.sleep(1)
    deleted_docs = loader.load()

    test_case.assertCountEqual(expected_docs, loaded_docs)
    assert len(deleted_docs) == 0


def test_firestore_empty_load():
    loader = DatastoreLoader("Empty")

    loaded_docs = loader.load()

    assert len(loaded_docs) == 0


def test_firestore_custom_client() -> None:
    client = Client(namespace="namespace")
    saver = DatastoreSaver("Custom", client=client)
    loader = DatastoreLoader("Custom", client=client)

    docs = [Document(page_content="data", metadata={})]

    saver.upsert_documents(docs)
    # wait 1s for consistency
    time.sleep(1)
    written_docs = loader.load()
    saver.delete_documents(written_docs)
    # wait 1s for consistency
    time.sleep(1)

    deleted_docs = loader.load()

    assert len(written_docs) == 1
    assert written_docs[0].page_content == "data"
    assert written_docs[0].metadata != {}
    assert "key" in written_docs[0].metadata
    assert len(deleted_docs) == 0
