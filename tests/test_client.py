from __future__ import annotations

import httpx
import pytest

from pydantic_ai_asyntai import AsyntaiClient, AsyntaiError, AsyntaiRequestError

from .conftest import FakeAsyntai, body

pytestmark = pytest.mark.anyio


def client(fake: FakeAsyntai, key: str = 'test-key') -> AsyntaiClient:
    return AsyntaiClient(key, base_url='https://api.test', transport=fake.transport)


def test_key_is_required() -> None:
    with pytest.raises(AsyntaiError, match='ASYNTAI_API_KEY'):
        AsyntaiClient()


def test_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ASYNTAI_API_KEY', 'env-key')
    assert AsyntaiClient().api_key == 'env-key'


def test_trailing_slash_is_removed() -> None:
    assert AsyntaiClient('k', base_url='https://api.test/').base_url == 'https://api.test'


async def test_chat_sends_message_session_and_website(fake: FakeAsyntai) -> None:
    payload = await client(fake).chat('Hi?', session_id='s9', website_id=7)
    request = fake.last('/api/v1/chat/')
    assert request.method == 'POST'
    assert request.headers['authorization'] == 'Bearer test-key'
    assert request.headers['user-agent'] == 'pydantic-ai-asyntai'
    assert body(request) == {'message': 'Hi?', 'session_id': 's9', 'website_id': 7}
    assert payload['response'] == 'Answer to: Hi?'


async def test_chat_leaves_out_empty_fields(fake: FakeAsyntai) -> None:
    await client(fake).chat('Hi?')
    assert body(fake.last('/api/v1/chat/')) == {'message': 'Hi?'}


async def test_list_endpoints_return_their_lists(fake: FakeAsyntai) -> None:
    c = client(fake)
    assert (await c.websites())[0]['domain'] == 'shop.test'
    assert (await c.knowledge(limit=5, website_id=7))[0]['title'] == 'Delivery'
    assert fake.last('/api/v1/knowledge/').url.params['limit'] == '5'
    assert fake.last('/api/v1/knowledge/').url.params['website_id'] == '7'
    leads = await c.leads(limit=3, since='2026-10-01')
    assert leads[0]['email'] == 'ana@example.test'
    assert fake.last('/api/v1/leads/').url.params['since'] == '2026-10-01'
    messages = await c.conversation('s1', limit=10)
    assert messages[0]['role'] == 'user'
    assert fake.last('/api/v1/conversations/').url.params['session_id'] == 's1'


async def test_write_endpoints(fake: FakeAsyntai) -> None:
    c = client(fake)
    await c.add_text('Hours', 'Open 9-17', website_id=7)
    assert body(fake.last('/api/v1/knowledge/text/')) == {'title': 'Hours', 'content': 'Open 9-17', 'website_id': 7}
    await c.add_url('https://shop.test/faq')
    assert body(fake.last('/api/v1/knowledge/url/')) == {'url': 'https://shop.test/faq'}


async def test_wrong_key(fake: FakeAsyntai) -> None:
    with pytest.raises(AsyntaiError, match='rejected the API key'):
        await client(fake, key='bad').websites()


@pytest.mark.parametrize(
    ('code', 'message'),
    [(403, 'Starter plan'), (429, 'message limit'), (500, 'fake error 500')],
)
async def test_errors_that_stop_the_run(fake: FakeAsyntai, code: int, message: str) -> None:
    fake.status['/api/v1/websites/'] = code
    with pytest.raises(AsyntaiError, match=message) as info:
        await client(fake).websites()
    assert not isinstance(info.value, AsyntaiRequestError)


@pytest.mark.parametrize('code', [400, 404])
async def test_errors_the_model_can_fix(fake: FakeAsyntai, code: int) -> None:
    fake.status['/api/v1/conversations/'] = code
    with pytest.raises(AsyntaiRequestError, match=f'fake error {code}'):
        await client(fake).conversation('missing')


async def test_network_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused', request=request)

    c = AsyntaiClient('k', base_url='https://api.test', transport=httpx.MockTransport(boom))
    with pytest.raises(AsyntaiError, match='Could not reach'):
        await c.websites()


async def test_non_json_body() -> None:
    c = AsyntaiClient(
        'k', base_url='https://api.test', transport=httpx.MockTransport(lambda r: httpx.Response(502, text='<html>'))
    )
    with pytest.raises(AsyntaiError, match='non-JSON'):
        await c.websites()
