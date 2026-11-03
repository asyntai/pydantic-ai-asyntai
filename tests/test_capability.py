from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic_ai import Agent, AgentSpec
from pydantic_ai.agent.spec import CapabilitySpec
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_asyntai import INSTRUCTIONS, AsyntaiCapability, AsyntaiError

from .conftest import FakeAsyntai, body

pytestmark = pytest.mark.anyio

READ_TOOLS = {'asyntai_ask_website', 'asyntai_list_websites', 'asyntai_list_knowledge'}
LEAD_TOOLS = {'asyntai_get_leads', 'asyntai_get_conversation'}
WRITE_TOOLS = {'asyntai_add_text', 'asyntai_add_url'}


def capability(fake: FakeAsyntai, **kwargs: Any) -> AsyntaiCapability:
    return AsyntaiCapability(api_key='test-key', base_url='https://api.test', transport=fake.transport, **kwargs)


def tool_names(agent: Agent[Any, Any]) -> set[str]:
    model = TestModel(call_tools=[])
    agent.run_sync('hi', model=model)
    assert model.last_model_request_parameters is not None
    return {t.name for t in model.last_model_request_parameters.function_tools}


def script(*steps: list[ToolCallPart] | str):
    """A FunctionModel that plays fixed tool calls, then a final text."""
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        step = steps[len(seen) - 1]
        if isinstance(step, str):
            return ModelResponse(parts=[TextPart(step)])
        return ModelResponse(parts=step)

    return FunctionModel(respond), seen


def tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]


def test_default_tools(fake: FakeAsyntai) -> None:
    agent = Agent(TestModel(), capabilities=[capability(fake)])
    assert tool_names(agent) == READ_TOOLS | LEAD_TOOLS


def test_knowledge_tools_are_opt_in(fake: FakeAsyntai) -> None:
    agent = Agent(TestModel(), capabilities=[capability(fake, knowledge_tools=True)])
    assert tool_names(agent) == READ_TOOLS | LEAD_TOOLS | WRITE_TOOLS


def test_lead_tools_can_be_turned_off(fake: FakeAsyntai) -> None:
    agent = Agent(TestModel(), capabilities=[capability(fake, lead_tools=False)])
    assert tool_names(agent) == READ_TOOLS


def test_instructions_reach_the_model(fake: FakeAsyntai) -> None:
    seen: list[str | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages[-1].instructions)
        return ModelResponse(parts=[TextPart('ok')])

    Agent(FunctionModel(respond), capabilities=[capability(fake)]).run_sync('hi')
    assert seen[0] is not None and INSTRUCTIONS in seen[0]


def test_every_tool_runs_against_the_api(fake: FakeAsyntai) -> None:
    agent = Agent(TestModel(), capabilities=[capability(fake, knowledge_tools=True, website_id=7)])
    agent.run_sync('Use every tool.')
    called = {r.url.path for r in fake.requests}
    assert called == {
        '/api/v1/chat/',
        '/api/v1/websites/',
        '/api/v1/knowledge/',
        '/api/v1/knowledge/text/',
        '/api/v1/knowledge/url/',
        '/api/v1/leads/',
        '/api/v1/conversations/',
    }
    assert body(fake.last('/api/v1/chat/'))['website_id'] == 7


def test_ask_returns_the_answer_and_keeps_the_session(fake: FakeAsyntai) -> None:
    model, seen = script(
        [ToolCallPart('asyntai_ask_website', {'question': 'Do you ship to Spain?'}, tool_call_id='a')],
        [ToolCallPart('asyntai_ask_website', {'question': 'How much does it cost?'}, tool_call_id='b')],
        'done',
    )
    result = Agent(model, capabilities=[capability(fake)]).run_sync('Ask twice.')
    returns = tool_returns(result.all_messages())
    assert [r.content for r in returns] == ['Answer to: Do you ship to Spain?', 'Answer to: How much does it cost?']
    chat_bodies = [body(r) for r in fake.requests if r.url.path == '/api/v1/chat/']
    first = chat_bodies[0]['session_id']
    assert first.startswith('pydantic-ai-') and len(first) > 20
    assert chat_bodies[1]['session_id'] == first


def test_runs_do_not_share_a_session(fake: FakeAsyntai) -> None:
    cap = capability(fake)
    for _ in range(2):
        model, _ = script(
            [ToolCallPart('asyntai_ask_website', {'question': 'Hi?'}, tool_call_id='a')],
            'done',
        )
        Agent(model, capabilities=[cap]).run_sync('Ask.')
    sessions = [body(r)['session_id'] for r in fake.requests if r.url.path == '/api/v1/chat/']
    assert len(set(sessions)) == 2
    assert cap.session_id is None


def test_a_fixed_session_continues_across_runs(fake: FakeAsyntai) -> None:
    cap = capability(fake, session_id='customer-42')
    for _ in range(2):
        model, _ = script(
            [ToolCallPart('asyntai_ask_website', {'question': 'Hi?'}, tool_call_id='a')],
            'done',
        )
        Agent(model, capabilities=[cap]).run_sync('Ask.')
    sessions = [body(r)['session_id'] for r in fake.requests if r.url.path == '/api/v1/chat/']
    assert sessions == ['customer-42', 'customer-42']


async def test_toolset_used_outside_a_run_still_gets_its_own_session(fake: FakeAsyntai) -> None:
    toolset = capability(fake).get_toolset()
    tool = toolset.tools['asyntai_ask_website']  # type: ignore[union-attr]
    await tool.function(question='Hi?')  # type: ignore[call-arg]
    await tool.function(question='And?')  # type: ignore[call-arg]
    sessions = [body(r)['session_id'] for r in fake.requests if r.url.path == '/api/v1/chat/']
    assert sessions[0] == sessions[1] and sessions[0].startswith('pydantic-ai-')


def test_same_agent_two_runs_do_not_share_a_session(fake: FakeAsyntai) -> None:
    calls = {'n': 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['n'] += 1
        if calls['n'] % 2 == 1:
            return ModelResponse(parts=[ToolCallPart('asyntai_ask_website', {'question': 'Hi?'})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[capability(fake)])
    agent.run_sync('one')
    agent.run_sync('two')
    sessions = [body(r)['session_id'] for r in fake.requests if r.url.path == '/api/v1/chat/']
    assert len(sessions) == 2 and sessions[0] != sessions[1]


def test_limits_are_clamped(fake: FakeAsyntai) -> None:
    model, _ = script(
        [
            ToolCallPart('asyntai_get_leads', {'limit': 5000}, tool_call_id='a'),
            ToolCallPart('asyntai_list_knowledge', {'limit': -3}, tool_call_id='b'),
        ],
        'done',
    )
    Agent(model, capabilities=[capability(fake)]).run_sync('Read.')
    assert fake.last('/api/v1/leads/').url.params['limit'] == '100'
    assert fake.last('/api/v1/knowledge/').url.params['limit'] == '1'


def test_empty_results_say_so_in_words(fake: FakeAsyntai, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    original = fake.handler

    def empty(request: httpx.Request) -> httpx.Response:
        keys = {'/api/v1/leads/': 'leads', '/api/v1/knowledge/': 'entries', '/api/v1/conversations/': 'messages'}
        if request.url.path in keys:
            fake.requests.append(request)
            return httpx.Response(200, json={'success': True, keys[request.url.path]: []})
        return original(request)

    monkeypatch.setattr(fake, 'handler', empty)
    model, _ = script(
        [
            ToolCallPart('asyntai_get_leads', {'since': '2026-10-01'}, tool_call_id='a'),
            ToolCallPart('asyntai_list_knowledge', {}, tool_call_id='b'),
            ToolCallPart('asyntai_get_conversation', {'session_id': 's7'}, tool_call_id='c'),
        ],
        'done',
    )
    result = Agent(model, capabilities=[capability(fake)]).run_sync('Read.')
    contents = {r.tool_name: r.content for r in tool_returns(result.all_messages())}
    assert contents['asyntai_get_leads'] == (
        'No website visitor left an email address or a phone number since 2026-10-01. The call worked.'
    )
    assert contents['asyntai_list_knowledge'] == 'The knowledge base of this website agent is empty.'
    assert contents['asyntai_get_conversation'] == 'The chat s7 has no messages.'


def test_a_bad_argument_lets_the_model_retry(fake: FakeAsyntai) -> None:
    fake.status['/api/v1/conversations/'] = 404
    model, seen = script(
        [ToolCallPart('asyntai_get_conversation', {'session_id': 'nope'}, tool_call_id='a')],
        'I could not find that chat.',
    )
    result = Agent(model, capabilities=[capability(fake)]).run_sync('Read chat nope.')
    assert result.output == 'I could not find that chat.'
    retry_text = json.dumps([p.model_dump() if hasattr(p, 'model_dump') else str(p) for p in seen[1][-1].parts], default=str)
    assert 'fake error 404' in retry_text


def test_a_wrong_key_stops_the_run(fake: FakeAsyntai) -> None:
    cap = AsyntaiCapability(api_key='wrong', base_url='https://api.test', transport=fake.transport)
    model, _ = script([ToolCallPart('asyntai_list_websites', {}, tool_call_id='a')], 'done')
    with pytest.raises(AsyntaiError, match='rejected the API key'):
        Agent(model, capabilities=[cap]).run_sync('List.')


def test_no_key_anywhere_stops_the_run(fake: FakeAsyntai) -> None:
    cap = AsyntaiCapability(base_url='https://api.test', transport=fake.transport)
    model, _ = script([ToolCallPart('asyntai_list_websites', {}, tool_call_id='a')], 'done')
    with pytest.raises(AsyntaiError, match='ASYNTAI_API_KEY'):
        Agent(model, capabilities=[cap]).run_sync('List.')


def test_key_from_environment(fake: FakeAsyntai, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ASYNTAI_API_KEY', 'test-key')
    cap = AsyntaiCapability(base_url='https://api.test', transport=fake.transport)
    model, _ = script([ToolCallPart('asyntai_list_websites', {}, tool_call_id='a')], 'done')
    Agent(model, capabilities=[cap]).run_sync('List.')
    assert fake.last('/api/v1/websites/').headers['authorization'] == 'Bearer test-key'


def test_loads_from_an_agent_spec(fake: FakeAsyntai, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ASYNTAI_API_KEY', 'test-key')
    agent = Agent.from_spec(
        AgentSpec(
            model='test',
            capabilities=[
                CapabilitySpec(name='AsyntaiCapability', arguments={'website_id': 7, 'knowledge_tools': True})
            ],
        ),
        custom_capability_types=[AsyntaiCapability],
    )
    assert tool_names(agent) == READ_TOOLS | LEAD_TOOLS | WRITE_TOOLS


def test_two_accounts_on_one_agent(fake: FakeAsyntai) -> None:
    first = capability(fake, website_id=1)
    second = capability(fake, website_id=2).prefix_tools('second')
    agent = Agent(TestModel(), capabilities=[first, second])
    names = tool_names(agent)
    assert 'asyntai_ask_website' in names
    assert 'second_asyntai_ask_website' in names
