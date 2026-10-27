"""Test doubles shared by more than one test module."""

from collections.abc import Sequence
from typing import Self

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ScriptedChatModel(GenericFakeChatModel):
    """A chat model that replies with scripted messages and ignores the schema it is given."""

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> Self:
        """Ignore the structured-output binding: the script already holds the reply."""
        return self
