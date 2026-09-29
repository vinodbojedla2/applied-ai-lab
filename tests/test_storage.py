from ai_lab.storage import DocumentStore


def test_sync_updates_and_deletes_documents(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "first.md"
    first.write_text("first version", encoding="utf-8")
    store = DocumentStore(tmp_path / "index.sqlite")
    assert store.sync(docs) == {"updated": 1, "deleted": 0, "passages": 1}
    assert store.sync(docs)["updated"] == 0
    first.write_text("second version", encoding="utf-8")
    assert store.sync(docs)["updated"] == 1
    assert store.passages()[0].text == "second version"
    first.unlink()
    assert store.sync(docs) == {"updated": 0, "deleted": 1, "passages": 0}
