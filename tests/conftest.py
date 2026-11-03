from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest


@dataclass
class FakeAsyntai:
    """A fake Asyntai API that records every request."""

    requests: list[httpx.Request] = field(default_factory=list)
    status: dict[str, int] = field(default_factory=dict)
    sessions: list[str] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.status:
            code = self.status[path]
            return httpx.Response(code, json={'success': False, 'error': f'fake error {code}'})
        if request.headers.get('authorization') != 'Bearer test-key':
            return httpx.Response(401, json={'success': False, 'error': 'Invalid API key'})
        if path == '/api/v1/chat/':
            body = json.loads(request.content)
            session = body.get('session_id') or f'api_sess_{len(self.sessions) + 1}'
            self.sessions.append(session)
            return httpx.Response(
                200,
                json={'success': True, 'response': f'Answer to: {body["message"]}', 'session_id': session},
            )
        if path == '/api/v1/websites/':
            return httpx.Response(200, json={'success': True, 'websites': [{'id': 7, 'domain': 'shop.test'}]})
        if path == '/api/v1/knowledge/':
            return httpx.Response(200, json={'success': True, 'entries': [{'id': 'k1', 'title': 'Delivery'}]})
        if path in ('/api/v1/knowledge/text/', '/api/v1/knowledge/url/'):
            return httpx.Response(201, json={'success': True, 'id': 'k2'})
        if path == '/api/v1/leads/':
            return httpx.Response(
                200,
                json={'success': True, 'leads': [{'session_id': 's1', 'email': 'ana@example.test', 'phone': None}]},
            )
        if path == '/api/v1/conversations/':
            return httpx.Response(
                200, json={'success': True, 'messages': [{'role': 'user', 'content': 'Do you ship to Spain?'}]}
            )
        return httpx.Response(404, json={'success': False, 'error': 'Not found'})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def last(self, path: str) -> httpx.Request:
        return [r for r in self.requests if r.url.path == path][-1]


@pytest.fixture
def fake() -> FakeAsyntai:
    return FakeAsyntai()


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ASYNTAI_API_KEY', raising=False)


def body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)
