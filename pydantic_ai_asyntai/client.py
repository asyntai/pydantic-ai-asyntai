"""Async HTTP client for the Asyntai public API.

Every tool in this package goes through this client, so authentication,
timeouts and error messages have one implementation. The API is documented at
https://asyntai.com/documentation/api-reference/.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = 'https://asyntai.com'
DEFAULT_TIMEOUT = 60.0
USER_AGENT = 'pydantic-ai-asyntai'


class AsyntaiError(RuntimeError):
    """The Asyntai API refused a call for a reason the model cannot fix.

    Examples: a wrong API key, a plan without API access, or the monthly
    message limit. These stop the agent run.
    """


class AsyntaiRequestError(AsyntaiError):
    """The Asyntai API refused the arguments of one call (HTTP 400 or 404).

    The capability turns this into ``ModelRetry`` so the model can correct the
    call, for example a wrong ``website_id`` or ``session_id``.
    """


def resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get('ASYNTAI_API_KEY', '')
    if not key:
        raise AsyntaiError(
            'An Asyntai API key is required. Pass api_key=... or set the '
            'ASYNTAI_API_KEY environment variable. Create a key in the Asyntai '
            'dashboard under Settings -> API.'
        )
    return key


class AsyntaiClient:
    """Calls the Asyntai REST API with one API key."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._transport = transport

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'User-Agent': USER_AGENT,
        }
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport,
            ) as http:
                response = await http.request(method, path, headers=headers, params=params, json=json_body)
        except httpx.HTTPError as exc:
            raise AsyntaiError(f'Could not reach the Asyntai API: {exc}') from exc

        try:
            payload = response.json()
        except ValueError:
            raise AsyntaiError(f'Asyntai returned a non-JSON response (HTTP {response.status_code}).') from None
        if not isinstance(payload, dict):
            raise AsyntaiError('Asyntai returned an unexpected response body.')

        if response.status_code == 401:
            raise AsyntaiError('Asyntai rejected the API key. Check the key in Settings -> API.')
        if response.status_code == 403:
            raise AsyntaiError('The Asyntai API needs the Starter plan or higher. See https://asyntai.com/pricing/.')
        if response.status_code == 429:
            raise AsyntaiError('The Asyntai message limit for this month is reached.')
        message = payload.get('error') or f'Asyntai API error (HTTP {response.status_code}).'
        if response.status_code in (400, 404):
            raise AsyntaiRequestError(message)
        if not payload.get('success', False):
            raise AsyntaiError(message)
        return payload

    async def chat(
        self,
        message: str,
        *,
        session_id: str | None = None,
        website_id: int | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/chat/: ask the website agent a question."""
        body: dict[str, Any] = {'message': message}
        if session_id:
            body['session_id'] = session_id
        if website_id is not None:
            body['website_id'] = website_id
        return await self._request('POST', '/api/v1/chat/', json_body=body)

    async def websites(self) -> list[dict[str, Any]]:
        """GET /api/v1/websites/: the website agents on this account."""
        return (await self._request('GET', '/api/v1/websites/')).get('websites', [])

    async def knowledge(self, *, limit: int = 50, website_id: int | None = None) -> list[dict[str, Any]]:
        """GET /api/v1/knowledge/: what the website agent knows."""
        params: dict[str, Any] = {'limit': limit}
        if website_id is not None:
            params['website_id'] = website_id
        return (await self._request('GET', '/api/v1/knowledge/', params=params)).get('entries', [])

    async def add_text(self, title: str, content: str, *, website_id: int | None = None) -> dict[str, Any]:
        """POST /api/v1/knowledge/text/: store a text in the knowledge base."""
        body: dict[str, Any] = {'title': title, 'content': content}
        if website_id is not None:
            body['website_id'] = website_id
        return await self._request('POST', '/api/v1/knowledge/text/', json_body=body)

    async def add_url(self, url: str, *, website_id: int | None = None) -> dict[str, Any]:
        """POST /api/v1/knowledge/url/: read a page into the knowledge base."""
        body: dict[str, Any] = {'url': url}
        if website_id is not None:
            body['website_id'] = website_id
        return await self._request('POST', '/api/v1/knowledge/url/', json_body=body)

    async def leads(
        self,
        *,
        limit: int = 20,
        website_id: int | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """GET /api/v1/leads/: visitors who left an email address or a phone number."""
        params: dict[str, Any] = {'limit': limit}
        if website_id is not None:
            params['website_id'] = website_id
        if since:
            params['since'] = since
        return (await self._request('GET', '/api/v1/leads/', params=params)).get('leads', [])

    async def conversation(self, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """GET /api/v1/conversations/: the messages of one website chat."""
        params = {'session_id': session_id, 'limit': limit}
        return (await self._request('GET', '/api/v1/conversations/', params=params)).get('messages', [])
