import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from leadfinder import auth
from leadfinder.service.app import create_app
from leadfinder.writer import Claude, Offline


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path / "test.db", demo=True, writer=Offline())


class FakeAnthropic:
    """Stands in for anthropic.Anthropic: answers `beta.messages.create` with the next queued reply and keeps the
    request, so a test can read what would have been sent."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):                      # a stop reason with no content, e.g. "refusal"
            return SimpleNamespace(stop_reason=reply, content=[])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="thinking", thinking=""),
                                                                SimpleNamespace(type="text", text=json.dumps(reply))])


@pytest.fixture
def fake():
    return FakeAnthropic()


@pytest.fixture
def claude_app(tmp_path, fake):
    return create_app(tmp_path / "claude.db", demo=True, writer=Claude(client=fake))


@pytest.fixture
def client(app):
    return TestClient(app)


def login(client, username="riley", password=auth.DEMO_PASSWORD):
    r = client.post("/login", data={"username": username, "password": password, "next": "/"}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return client


@pytest.fixture
def sdr(client):
    return login(client, "riley")


@pytest.fixture
def manager(app):
    return login(TestClient(app), "morgan")


def form_value(html: str, name: str) -> str:
    return re.search(rf'name="{name}" value="([^"]*)"', html).group(1)
