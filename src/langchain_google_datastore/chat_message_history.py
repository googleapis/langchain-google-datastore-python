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

import json
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

from google.cloud import datastore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

from .version import __version__

USER_AGENT = "langchain-google-datastore-python:chat_history" + __version__
DEFAULT_KIND = "ChatHistory"

if TYPE_CHECKING:
    from google.cloud.datastore import Client  # type: ignore


class DatastoreChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        kind: str = DEFAULT_KIND,
        client: Optional[Client] = None,
    ) -> None:
        """Chat Message History for Google Cloud Firestore.
        Args:
            session_id: Arbitrary key that is used to store the messages of a single
                chat session. This is the identifier of an entity.
            kind: The name of the Datastore kind to write into. This is an optional value
                and by default it will use `ChatHistory` as the kind.
            client: Client for interacting with the Google Cloud Firestore API.
        """
        self.client = client or datastore.Client()
        client_agent = self.client._client_info.user_agent
        if not client_agent:
            self.client._client_info.user_agent = USER_AGENT
        elif USER_AGENT not in client_agent:
            self.client._client_info.user_agent = " ".join([client_agent, USER_AGENT])
        self.session_id = session_id
        self.key = self.client.key(kind, session_id)
        self.messages: List[BaseMessage] = []
        self._load_messages()

    def _load_messages(self) -> None:
        entity = self.client.get(self.key)
        if entity:
            data_entity = dict(entity.items())
            if "messages" in data_entity:
                self.messages = decode_messages(data_entity["messages"])

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self._upsert_messages()

    def _upsert_messages(self) -> None:
        entity = self.client.entity(self.key)
        entity["messages"] = encode_messages(self.messages)
        self.client.put(entity)

    def clear(self) -> None:
        self.messages = []
        self.client.delete(self.key)


def encode_messages(messages: List[BaseMessage]) -> List[bytes]:
    return [str.encode(m.json()) for m in messages]


def decode_messages(messages: List[bytes]) -> List[BaseMessage]:
    dict_messages = [json.loads(m.decode()) for m in messages]
    return messages_from_dict([{"type": m["type"], "data": m} for m in dict_messages])
