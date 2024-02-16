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
import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Dict, List

import more_itertools
from google.cloud.datastore import Entity, Key
from google.cloud.datastore.helpers import GeoPoint
from langchain_core.documents import Document

if TYPE_CHECKING:
    from google.cloud.datastore import Client, Entity


DATASTORE_TYPE = "datastore_type"
KEY = "key"
ENTITY = "entity"
GEOPOINT = "geopoint"


def convert_firestore_entity(
    entity: Entity,
    page_content_properties: List[str] = [],
    metadata_properties: List[str] = [],
) -> Document:
    data_entity = dict(entity.items())
    metadata = {
        "key": {
            "path": entity.key.flat_path,
            DATASTORE_TYPE: KEY,
        }
    }

    set_page_properties = set(
        page_content_properties or (data_entity.keys() - set(metadata_properties))
    )
    set_metadata_properties = set(
        metadata_properties or (data_entity.keys() - set_page_properties)
    )

    page_content = {}

    for k in sorted(set_metadata_properties):
        if k in data_entity:
            metadata[k] = _convert_from_firestore(data_entity[k])
    for k in sorted(set_page_properties):
        if k in data_entity:
            print("--here")
            page_content[k] = _convert_from_firestore(data_entity[k])
            print(page_content)
            print("--here")

    if len(page_content) == 1:
        page_content = str(page_content.popitem()[1])  # type: ignore
    else:
        page_content = json.dumps(page_content)  # type: ignore

    print("--after conversion hhere")
    print(page_content)
    print("--after conversion hhere")
    doc = Document(page_content=page_content, metadata=metadata)  # type: ignore
    return doc


def convert_langchain_document(document: Document, client: Client) -> dict:
    metadata = document.metadata.copy()
    path = None
    data = {}

    if metadata.get("key", {}).get(DATASTORE_TYPE) == KEY:
        path = metadata["key"]
        metadata.pop("key")

    if metadata:
        data.update(_convert_from_langchain(metadata, client))

    if document.page_content:
        try:
            content_dict = json.loads(document.page_content)
        except (ValueError, SyntaxError):
            content_dict = {"page_content": document.page_content}
        converted_page = _convert_from_langchain(content_dict, client)
        data.update(converted_page)

    return {"key": path, "properties": data}


def _convert_from_firestore(val: Any) -> Any:
    val_converted = val
    if isinstance(val, dict):
        val_converted = {k: _convert_from_firestore(v) for k, v in val.items()}
    if isinstance(val, list):
        val_converted = [_convert_from_firestore(v) for v in val]
    elif isinstance(val, Key):
        val_converted = {
            "key": val.flat_path,
            DATASTORE_TYPE: KEY,
        }
    elif isinstance(val, GeoPoint):
        val_converted = {
            "latitude": val.latitude,
            "longitude": val.longitude,
            DATASTORE_TYPE: GEOPOINT,
        }
    elif isinstance(val, Entity):
        val_converted = {
            "key": val.key.flat_path,
            "properties": _convert_from_firestore(dict(val.items())),
            DATASTORE_TYPE: ENTITY,
        }

    return val_converted


def _convert_from_langchain(val: Any, client: Client) -> Any:
    val_converted = val
    if isinstance(val, list):
        val_converted = [_convert_from_langchain(v, client) for v in val]
    elif isinstance(val, dict):
        l = len(val)
        if val.get(DATASTORE_TYPE) == KEY:
            val_converted = client.key(*val["key"])
        elif val.get(DATASTORE_TYPE) == GEOPOINT:
            val_converted = GeoPoint(val["latitude"], val["longitude"])
        elif val.get(DATASTORE_TYPE) == ENTITY:
            key = client.key(*val["key"])
            entity = client.entity(key)
            entity.update(val["properties"])
            val_converted = entity
        else:
            val_converted = {
                k: _convert_from_langchain(v, client) for k, v in val.items()
            }
    return val_converted
