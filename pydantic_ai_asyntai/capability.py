"""The Asyntai capability for Pydantic AI agents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from .client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, AsyntaiClient, AsyntaiRequestError

INSTRUCTIONS = (
    'You can reach the Asyntai website agent of this business. It answers from the '
    "business's own website, product catalog and documents, which Asyntai keeps current. "
    'For any question about the business (products, prices, stock, delivery, opening hours, '
    'policies, documentation), call `asyntai_ask_website` instead of guessing, and say that '
    'the answer comes from the website. Ask follow-up questions in the same run: the website '
    'agent keeps the context of the conversation.'
)


@dataclass
class AsyntaiCapability(AbstractCapability[Any]):
    """Gives an agent the knowledge, chats and leads of an Asyntai website agent.

    Asyntai reads a website, its product feed and its documents, and keeps that
    knowledge current. This capability hands it to a Pydantic AI agent as tools,
    so the agent answers questions about a real business without a local index.

    Args:
        api_key: Asyntai API key. Defaults to the ``ASYNTAI_API_KEY`` environment variable.
        website_id: Which website agent to use, when the account has several.
            ``asyntai_list_websites`` returns the ids.
        knowledge_tools: Also register ``asyntai_add_text`` and ``asyntai_add_url``,
            which write to the knowledge base. Off by default.
        lead_tools: Register ``asyntai_get_leads`` and ``asyntai_get_conversation``,
            which read visitor contact details and chats. On by default.
        session_id: The website conversation to continue across runs. By default each
            run starts its own conversation, and questions in one run share it.
        base_url: API address. Change it only for a self-hosted test server.
        timeout: Seconds to wait for one API call.
    """

    api_key: str | None = None
    website_id: int | None = None
    knowledge_tools: bool = False
    lead_tools: bool = True
    session_id: str | None = None
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    id: str | None = None
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False, compare=False)

    def _client(self) -> AsyntaiClient:
        return AsyntaiClient(self.api_key, base_url=self.base_url, timeout=self.timeout, transport=self.transport)

    async def for_run(self, ctx: RunContext[Any]) -> AsyntaiCapability:
        # A fresh instance per run. Without a session id the Asyntai API puts every
        # call of the account into one shared conversation, so each run gets its own.
        return AsyntaiCapability(
            api_key=self.api_key,
            website_id=self.website_id,
            knowledge_tools=self.knowledge_tools,
            lead_tools=self.lead_tools,
            session_id=self.session_id or f'pydantic-ai-{uuid.uuid4().hex}',
            base_url=self.base_url,
            timeout=self.timeout,
            id=self.id,
            transport=self.transport,
        )

    def get_instructions(self) -> str:
        return INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[Any] | None:
        toolset: FunctionToolset[Any] = FunctionToolset()

        async def asyntai_ask_website(question: str) -> str:
            """Ask the website agent a question and get its answer.

            Use it for anything about the business: products, prices, stock, delivery,
            returns, opening hours, policies and documentation.

            Args:
                question: The question, in any language.
            """
            if not self.session_id:  # the toolset was used outside an agent run
                self.session_id = f'pydantic-ai-{uuid.uuid4().hex}'
            payload = await _call(self._client().chat(question, session_id=self.session_id, website_id=self.website_id))
            return str(payload.get('response', ''))

        async def asyntai_list_websites() -> list[dict[str, Any]] | str:
            """List the website agents on this Asyntai account, with their ids and domains."""
            websites = await _call(self._client().websites())
            return websites or 'This Asyntai account has no website agents yet.'

        async def asyntai_list_knowledge(limit: int = 50) -> list[dict[str, Any]] | str:
            """List the pages, texts and files the website agent knows.

            Args:
                limit: How many entries to return, 1 to 200.
            """
            entries = await _call(self._client().knowledge(limit=_clamp(limit, 1, 200), website_id=self.website_id))
            return entries or 'The knowledge base of this website agent is empty.'

        toolset.add_function(asyntai_ask_website)
        toolset.add_function(asyntai_list_websites)
        toolset.add_function(asyntai_list_knowledge)

        if self.lead_tools:

            async def asyntai_get_leads(limit: int = 20, since: str | None = None) -> list[dict[str, Any]] | str:
                """Read the website visitors who left an email address or a phone number.

                Args:
                    limit: How many leads to return, newest first, 1 to 100.
                    since: Only leads after this ISO date, for example 2026-10-01.
                """
                leads = await _call(
                    self._client().leads(limit=_clamp(limit, 1, 100), website_id=self.website_id, since=since)
                )
                if leads:
                    return leads
                period = f' since {since}' if since else ''
                return f'No website visitor left an email address or a phone number{period}. The call worked.'

            async def asyntai_get_conversation(session_id: str, limit: int = 50) -> list[dict[str, Any]] | str:
                """Read the messages of one website chat.

                Args:
                    session_id: The chat's session id, as returned with a lead.
                    limit: How many messages to return, 1 to 200.
                """
                messages = await _call(self._client().conversation(session_id, limit=_clamp(limit, 1, 200)))
                return messages or f'The chat {session_id} has no messages.'

            toolset.add_function(asyntai_get_leads)
            toolset.add_function(asyntai_get_conversation)

        if self.knowledge_tools:

            async def asyntai_add_text(title: str, content: str) -> str:
                """Store a text in the knowledge base, so the website agent uses it in later answers.

                Args:
                    title: A short title for the text.
                    content: The text itself.
                """
                await _call(self._client().add_text(title, content, website_id=self.website_id))
                return f'Stored "{title}" in the knowledge base.'

            async def asyntai_add_url(url: str) -> str:
                """Read a web page into the knowledge base.

                Args:
                    url: The full address of the page, starting with https://.
                """
                await _call(self._client().add_url(url, website_id=self.website_id))
                return f'Asyntai is reading {url} into the knowledge base.'

            toolset.add_function(asyntai_add_text)
            toolset.add_function(asyntai_add_url)

        return toolset


async def _call(awaitable: Any) -> Any:
    try:
        return await awaitable
    except AsyntaiRequestError as exc:
        raise ModelRetry(str(exc)) from exc


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
