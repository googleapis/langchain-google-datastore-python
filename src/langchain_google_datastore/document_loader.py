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

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

import more_itertools
from google.cloud import datastore
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from .document_converter import (
    DATASTORE_TYPE,
    KEY,
    convert_firestore_entity,
    convert_langchain_document,
)
from .version import __version__

USER_AGENT_LOADER = "langchain-google-datastore-python:document_loader" + __version__
USER_AGENT_SAVER = "langchain-google-datastore-python:document_saver" + __version__
WRITE_BATCH_SIZE = 500

if TYPE_CHECKING:
    from google.cloud.datastore import Client, Query


class DatastoreLoader(BaseLoader):
    def __init__(
        self,
        source: Query | str,
        page_content_properties: List[str] = [],
        metadata_properties: List[str] = [],
        client: Optional[Client] = None,
    ) -> None:
        """Document Loader for Google Cloud Firestore in Datastore Mode.
        Args:
            source: The source to load the documents. It can be an instance of Query
                or the name of the Datastore kind to read from.
            page_content_propeties: The properties to write into the `page_content`.
                If an empty or None list is provided all fields will be written into
                `page_content`. When only one field is provided only the value is written.
            metadata_properties: The properties to write into the `metadata`.
                By default it will write all fields that are not in `page_content` into `metadata`.
            client: Client for interacting with the Google Cloud Datastore API.
        """
        self.client = client_with_user_agent(client, USER_AGENT_LOADER)
        self.source = source
        self.page_content_properties = page_content_properties
        self.metadata_properties = metadata_properties

    def load(self) -> List[Document]:
        """Load Documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for Documents."""

        if isinstance(self.source, str):
            query = self.client.query(kind=self.source)
        else:
            query = self.source
            query._client = client_with_user_agent(query._client, USER_AGENT_LOADER)

        for entity in query.fetch():
            yield convert_firestore_entity(
                entity, self.page_content_properties, self.metadata_properties
            )


class DatastoreSaver:
    """Write into Google Cloud Platform `Firestore` in Datastore Mode."""

    def __init__(
        self,
        kind: Optional[str] = None,
        client: Optional[Client] = None,
    ) -> None:
        """Document Saver for Google Cloud Firestore in Datastore Mode.
        Args:
            kind: The kind to write the entities into. If this
              value is present it will write entities with an auto generated id.
            client: Client for interacting with the Google Cloud Datastore API.
        """
        self.kind = kind
        self.client = client_with_user_agent(client, USER_AGENT_SAVER)

    def upsert_documents(
        self,
        documents: List[Document],
    ) -> None:
        """Create / merge documents into the Firestore database in Datastore Mode.
        Args:
         documents: List of documents to be written into Datastore.
        """
        for batch in more_itertools.chunked(documents, WRITE_BATCH_SIZE):
            db_batch = self.client.batch()
            db_batch.begin()
            for doc in batch:
                entity_dict = convert_langchain_document(doc, self.client)
                if self.kind:
                    key = self.client.key(self.kind)
                elif entity_dict.get("key", {}).get(DATASTORE_TYPE) == KEY:
                    key = self.client.key(*entity_dict["key"]["path"])
                else:
                    raise ValueError(
                        "Unable to construct key for document: " + str(doc)
                    )
                entity = self.client.entity(key)
                entity.update(entity_dict["properties"])
                db_batch.put(entity)
            db_batch.commit()

    def delete_documents(
        self, documents: List[Document], keys: Optional[List[List[str]]] = None
    ) -> None:
        """Delete documents from the Datastore database.
        Args:
          documents: List of documents to be deleted from Datastore . It will try to create
            the entity key from the `key` in the document metadata.
          keys: List of Key paths to be delted from Datastore. If provided `documents` will
            be ignored.
        """
        docs_list = itertools.zip_longest(documents, keys or [])
        for batch in more_itertools.chunked(docs_list, WRITE_BATCH_SIZE):
            db_batch = self.client.batch()
            db_batch.begin()
            for doc, key_path in batch:
                key = None
                if key_path:
                    key = self.client.key(*key_path)
                elif doc:
                    entity_dict = convert_langchain_document(doc, self.client)
                    if entity_dict.get("key", {}).get(DATASTORE_TYPE) == KEY:
                        key = self.client.key(*entity_dict["key"]["path"])
                if not key:
                    raise ValueError(
                        "Unable to construct key for document: "
                        + str(doc)
                        + " or key: "
                        + str(key_path)
                    )
                db_batch.delete(key)
            db_batch.commit()


def client_with_user_agent(client: Client | None, user_agent: str) -> Client:
    if not client:
        client = datastore.Client()
    client_agent = client._client_info.user_agent
    if not client_agent:
        client._client_info.user_agent = user_agent
    elif user_agent not in client_agent:
        client._client_info.user_agent = " ".join([client_agent, user_agent])
    return client
