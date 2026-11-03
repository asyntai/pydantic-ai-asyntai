# pydantic-ai-asyntai

A [Pydantic AI](https://ai.pydantic.dev) capability for [Asyntai](https://asyntai.com), the AI chat agent for websites.

Point Asyntai at a website. It reads the pages, the product feed and the documents, and keeps them current. `AsyntaiCapability` hands that knowledge to any Pydantic AI agent, together with the website's chats and leads. Your agent answers questions about a real business without you building or maintaining an index.

Full guide: [asyntai.com/documentation/integrations/pydantic-ai](https://asyntai.com/documentation/integrations/pydantic-ai/)

## Install

```bash
pip install pydantic-ai-asyntai
```

## Set the key

Create an API key in the Asyntai dashboard under **Settings → API**. The API is available on the Starter plan and higher.

```bash
export ASYNTAI_API_KEY="your-key"
```

## Use it

```python
from pydantic_ai import Agent
from pydantic_ai_asyntai import AsyntaiCapability

agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[AsyntaiCapability()])

result = agent.run_sync('Do they ship to Spain, and how long does it take?')
print(result.output)
```

The capability adds tools and a short instruction that tells the model to ask the website instead of guessing. Each run starts its own website conversation, and follow-up questions in the run share its context. To continue one conversation across runs, for example one per customer, pass `session_id="customer-42"`. Every conversation also shows in your Asyntai chat logs.

## The tools

| Tool | What the agent can do | On by default |
| --- | --- | --- |
| `asyntai_ask_website` | Answer a question from the website: products, prices, stock, delivery, opening hours, policies, documentation. | Yes |
| `asyntai_list_websites` | List the website agents on the account, with their ids. | Yes |
| `asyntai_list_knowledge` | List the pages, texts and files the website agent knows. | Yes |
| `asyntai_get_leads` | Read the visitors who left an email address or a phone number. | Yes (`lead_tools`) |
| `asyntai_get_conversation` | Read the messages of one website chat. | Yes (`lead_tools`) |
| `asyntai_add_text` | Store a text so the website agent uses it in later answers. | No (`knowledge_tools`) |
| `asyntai_add_url` | Read a page into the knowledge base. | No (`knowledge_tools`) |

```python
AsyntaiCapability(
    website_id=42,          # which website agent, when the account has several
    lead_tools=False,       # hide visitor contact details and chats
    knowledge_tools=True,   # allow writing to the knowledge base
)
```

## Agent specs

The capability takes only plain values, so it works in [agent specs](https://ai.pydantic.dev/agent-spec/):

```yaml
model: anthropic:claude-sonnet-4-5
capabilities:
  - AsyntaiCapability: {website_id: 42}
```

```python
from pydantic_ai import Agent
from pydantic_ai_asyntai import AsyntaiCapability

agent = Agent.from_file('agent.yaml', custom_capability_types=[AsyntaiCapability])
```

## Several websites on one agent

An agency runs one Asyntai account with one website agent per client. Give each capability its own `website_id` and a tool prefix:

```python
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[
        AsyntaiCapability(website_id=1).prefix_tools('bakery'),
        AsyntaiCapability(website_id=2).prefix_tools('bikeshop'),
    ],
)
```

## Errors

A wrong argument, such as an unknown chat `session_id`, comes back to the model as `ModelRetry`, so the model can correct the call. A wrong API key, a plan without API access, or a reached message limit raises `AsyntaiError` and stops the run with a message you can show to a user.

## Tests

```bash
pip install -e ".[test]"
pytest
```

`tests/test_live.py` runs against a real account when `ASYNTAI_LIVE_KEY` and `ASYNTAI_LIVE_URL` are set.

## License

MIT
