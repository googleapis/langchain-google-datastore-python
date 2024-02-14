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

import ast
import itertools
from typing import TYPE_CHECKING, Any, Dict, List

import more_itertools
from google.cloud.datastore import Entity, Key
from google.cloud.datastore.helpers import GeoPoint
from langchain_core.documents import Document

if TYPE_CHECKING:
    from google.cloud.datastore import Client, Entity


class DocumentConverter:
    @staticmethod
    def convertFirestoreEntity(
        entity: Entity,
        page_content_properties: List[str] = [],
        metadata_properties: List[str] = [],
    ) -> Document:
        data_entity = dict(entity.items())
        metadata = {"key": entity.key.flat_path}

        set_page_properties = set(page_content_properties or [])
        set_metadata_properties = set(metadata_properties or [])
        shared_keys = set_metadata_properties & set_page_properties

        page_content = {}
        for k in sorted(shared_keys):
            if k in data_entity:
                val = DocumentConverter._convertFromFirestore(data_entity.pop(k))
                page_content[k] = val
                metadata[k] = val

        metadata.update(
            {
                k: DocumentConverter._convertFromFirestore(data_entity.pop(k))
                for k in sorted(set_metadata_properties - shared_keys)
                if k in data_entity
            }
        )
        if not set_page_properties:
            # write all properties
            keys = sorted(data_entity.keys())
            page_content = {
                k: DocumentConverter._convertFromFirestore(data_entity.pop(k))
                for k in keys
            }
        else:
            page_content.update(
                {
                    k: DocumentConverter._convertFromFirestore(data_entity.pop(k))
                    for k in sorted(set_page_properties - shared_keys)
                    if k in data_entity
                }
            )

        if len(page_content) == 1:
            page_content = page_content.popitem()[1]
        elif not page_content:
            page_content = ""  # type: ignore

        if not set_metadata_properties:
            # metadata fields not specified. Write remaining fields into metadata
            metadata.update(
                {
                    k: DocumentConverter._convertFromFirestore(v)
                    for k, v in sorted(data_entity.items())
                }
            )

        doc = Document(page_content=str(page_content), metadata=metadata)
        return doc

    @staticmethod
    def convertLangChainDocument(document: Document, client: Client) -> dict:
        metadata = document.metadata.copy()
        path = None
        data = {}

        if ("key" in metadata) and isinstance(metadata["key"], tuple):
            path = metadata["key"]
            metadata.pop("key")

        if metadata:
            data.update(DocumentConverter._convertFromLangChain(metadata, client))

        if document.page_content:
            try:
                content_dict = ast.literal_eval(document.page_content)
            except (ValueError, SyntaxError):
                content_dict = {"page_content": document.page_content}
            converted_page = DocumentConverter._convertFromLangChain(
                content_dict, client
            )
            data.update(converted_page)

        return {"key": path, "properties": data}

    @staticmethod
    def _convertFromFirestore(val: Any) -> Any:
        val_converted = val
        if isinstance(val, Key):
            val_converted = {"key": val.flat_path}
        elif isinstance(val, GeoPoint):
            val_converted = {"latitude": val.latitude, "longitude": val.longitude}
        elif isinstance(val, Entity):
            val_converted = {
                "key": val.key.flat_path,
                "properties": DocumentConverter._convertFromFirestore(
                    dict(val.items())
                ),
            }
        elif isinstance(val, dict):
            val_converted = {
                k: DocumentConverter._convertFromFirestore(v) for k, v in val.items()
            }
        elif isinstance(val, list):
            val_converted = [DocumentConverter._convertFromFirestore(v) for v in val]

        return val_converted

    @staticmethod
    def _convertFromLangChain(val: Any, client: Client) -> Any:
        val_converted = val
        if isinstance(val, dict):
            l = len(val)
            if (l == 1) and ("key" in val) and isinstance(val["key"], tuple):
                val_converted = client.key(*val["key"])
            elif (l == 2) and ("latitude" in val) and ("longitude" in val):
                val_converted = GeoPoint(val["latitude"], val["longitude"])
            elif (l == 2) and ("key" in val) and ("properties" in val):
                key = client.key(*val["key"])
                entity = client.entity(key)
                entity.update(val["properties"])
                val_converted = entity
            else:
                val_converted = {
                    k: DocumentConverter._convertFromLangChain(v, client)
                    for k, v in val.items()
                }
        elif isinstance(val, list):
            val_converted = [
                DocumentConverter._convertFromLangChain(v, client) for v in val
            ]
        return val_converted
