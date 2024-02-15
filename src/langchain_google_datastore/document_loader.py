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

from .utility.document_converter import DocumentConverter

USER_AGENT = "langchain-google-datastore-python"
WRITE_BATCH_SIZE = 500
TYPE = "datastore_type"

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
            client: Client for interacting with the Google Cloud Firestore API.
        """
        if client:
            self.client = client
            self.client._client_info.user_agent = USER_AGENT
        else:
            self.client = datastore.Client()
            self.client._client_info.user_agent = USER_AGENT

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
            query._client._client_info.user_agent = USER_AGENT

        for entity in query.fetch():
            yield DocumentConverter.convertFirestoreEntity(
                entity, self.page_content_properties, self.metadata_properties
            )


class DatastoreSaver:
    """Write into Google Cloud Platform `Firestore` in Datastore Mode."""

    def __init__(
        self,
        kind: Optional[str] = None,
        client: Optional[Client] = None,
    ) -> None:
        """Document Saver for Google Cloud Firestore.
        Args:
            kind: The kind to write the entities into. If this
              value is present it will write entities with an auto generated id.
            client: Client for interacting with the Google Cloud Firestore API.
        """
        self.kind = kind
        if client:
            self.client = client
            self.client._client_info.user_agent = USER_AGENT
        else:
            self.client = datastore.Client()
            self.client._client_info.user_agent = USER_AGENT

    def upsert_documents(
        self,
        documents: List[Document],
    ) -> None:
        """Create / merge documents into the Firestore database in Datastore Mode.
        Args:
         documents: List of documents to be written into Firestore.
        """
        for batch in more_itertools.chunked(documents, WRITE_BATCH_SIZE):
            db_batch = self.client.batch()
            db_batch.begin()
            for doc in batch:
                entity_dict = DocumentConverter.convertLangChainDocument(
                    doc, self.client
                )
                if self.kind:
                    key = self.client.key(self.kind)
                elif (
                    entity_dict["key"]
                    and ("type" in entity_dict["key"])
                    and (entity_dict["key"]["type"] == TYPE)
                ):
                    key = self.client.key(*entity_dict["key"]["path"])
                else:
                    continue
                entity = self.client.entity(key)
                entity.update(entity_dict["properties"])
                db_batch.put(entity)
            db_batch.commit()

    def delete_documents(
        self, documents: List[Document], keys: Optional[List[List[str]]] = None
    ) -> None:
        """Delete documents from the Firestore database.
        Args:
          documents: List of documents to be deleted from Firestore. It will try to create
            the entity key from the `key` in the document metadata.
          keys: List of Key paths to be delted from Firestore. If provided `documents` will
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
                    entity_dict = DocumentConverter.convertLangChainDocument(
                        doc, self.client
                    )
                    if (
                        entity_dict["key"]
                        and ("type" in entity_dict["key"])
                        and (entity_dict["key"]["type"] == TYPE)
                    ):
                        key = self.client.key(*entity_dict["key"]["path"])
                if not key:
                    continue
                db_batch.delete(key)
            db_batch.commit()
