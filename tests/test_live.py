"""Runs against a real Asyntai server. Skipped unless both variables are set.

    ASYNTAI_LIVE_KEY=...  ASYNTAI_LIVE_URL=https://asyntai.com  pytest tests/test_live.py
"""

from __future__ import annotations

import os

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_asyntai import AsyntaiCapability, AsyntaiClient, AsyntaiError

KEY = os.environ.get('ASYNTAI_LIVE_KEY')
URL = os.environ.get('ASYNTAI_LIVE_URL')
WEBSITE = os.environ.get('ASYNTAI_LIVE_WEBSITE_ID')

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not (KEY and URL), reason='set ASYNTAI_LIVE_KEY and ASYNTAI_LIVE_URL'),
]


def live_capability(**kwargs: object) -> AsyntaiCapability:
    return AsyntaiCapability(
        api_key=KEY, base_url=URL or '', website_id=int(WEBSITE) if WEBSITE else None, timeout=120, **kwargs
    )


async def test_websites_and_knowledge() -> None:
    client = AsyntaiClient(KEY, base_url=URL or '')
    assert isinstance(await client.websites(), list)
    assert isinstance(await client.knowledge(limit=5), list)


async def test_wrong_key_is_rejected() -> None:
    with pytest.raises(AsyntaiError, match='rejected the API key'):
        await AsyntaiClient('wrong-key', base_url=URL or '').websites()


def test_every_read_tool_through_an_agent() -> None:
    result = Agent(TestModel(), capabilities=[live_capability()]).run_sync('Use the tools.')
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    names = {p.tool_name for p in returns}
    assert {'asyntai_ask_website', 'asyntai_list_websites', 'asyntai_list_knowledge', 'asyntai_get_leads'} <= names
    answer = next(p.content for p in returns if p.tool_name == 'asyntai_ask_website')
    assert isinstance(answer, str) and answer.strip()
