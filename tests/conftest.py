import re

import pytest
from fastapi.testclient import TestClient

from leadfinder import auth
from leadfinder.service.app import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path / "test.db", demo=True)


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
