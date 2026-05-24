"""Tests for the FastAPI service (requires sentence-transformers + fastapi)."""

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from astrum_verum.api import create_app  # noqa: E402
from astrum_verum.engine import AstrumEngine  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    engine = AstrumEngine()
    app = create_app(engine)
    return TestClient(app)


class TestAPI:
    def test_lattice_info(self, client: TestClient) -> None:
        resp = client.get("/lattice/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "D4"
        assert data["dimension"] == 4
        assert data["num_vertices"] == 24
        assert data["num_edges"] == 96

    def test_add_memory(self, client: TestClient) -> None:
        resp = client.post("/memory/add", json={"text": "Test memory"})
        assert resp.status_code == 200
        assert "node_id" in resp.json()

    def test_add_memory_with_metadata(self, client: TestClient) -> None:
        resp = client.post(
            "/memory/add",
            json={"text": "Another memory", "metadata": {"tag": "test"}},
        )
        assert resp.status_code == 200

    def test_search(self, client: TestClient) -> None:
        client.post("/memory/add", json={"text": "Python programming"})
        resp = client.post(
            "/memory/search", json={"query": "coding", "top_k": 3}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_state(self, client: TestClient) -> None:
        resp = client.get("/memory/state")
        assert resp.status_code == 200
        body = resp.json()
        assert "lattice" in body
        assert "total_nodes" in body

    def test_cells(self, client: TestClient) -> None:
        resp = client.get("/memory/cells")
        assert resp.status_code == 200
        cells = resp.json()
        assert isinstance(cells, list)
        assert len(cells) == 24  # D₄

    def test_cell_nodes(self, client: TestClient) -> None:
        resp = client.get("/memory/cell/0/nodes")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
