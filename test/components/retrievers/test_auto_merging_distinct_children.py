# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.retrievers import AutoMergingRetriever, SentenceWindowRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("indices", "threshold", "merges"),
    [([0, 0, 0], 0.5, False), ([0, 1, 0, 1], 0.5, False), ([0, 1, 2], 0.5, True), ([2, 0, 1, 0], 0.75, False)],
)
async def test_merge_score_counts_distinct_children(
    indices: list[int], threshold: float, merges: bool, use_async: bool
) -> None:
    """Duplicate matches must not increase coverage or change unmerged output order."""
    hierarchy = HierarchicalDocumentSplitter(block_sizes={2}).run(
        [Document(content="one two three four five six seven eight")]
    )["documents"]
    leaves = [doc for doc in hierarchy if not doc.meta["__children_ids"]]
    store = InMemoryDocumentStore()
    store.write_documents([doc for doc in hierarchy if doc.meta["__children_ids"]])
    documents = [leaves[index] for index in indices]
    retriever = AutoMergingRetriever(document_store=store, threshold=threshold)

    result = await retriever.run_async(documents=documents) if use_async else retriever.run(documents=documents)

    assert result == {"documents": [hierarchy[0]] if merges else documents}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_overlapping_windows_do_not_inflate_parent_coverage(use_async: bool) -> None:
    """Real sentence-window output covers three, not five, of four child nodes."""
    hierarchy = HierarchicalDocumentSplitter(block_sizes={2}).run(
        [Document(content="one two three four five six seven eight")]
    )["documents"]
    leaves = [doc for doc in hierarchy if not doc.meta["__children_ids"]]
    leaf_store = InMemoryDocumentStore()
    leaf_store.write_documents(leaves)
    parent_store = InMemoryDocumentStore()
    parent_store.write_documents([doc for doc in hierarchy if doc.meta["__children_ids"]])
    windows = SentenceWindowRetriever(document_store=leaf_store, window_size=1)
    context = (
        await windows.run_async(retrieved_documents=leaves[:2])
        if use_async
        else windows.run(retrieved_documents=leaves[:2])
    )["context_documents"]
    assert [doc.meta["split_id"] for doc in context] == [0, 1, 0, 1, 2]
    assert len({doc.id for doc in context}) == 3
    retriever = AutoMergingRetriever(document_store=parent_store, threshold=0.8)

    result = await retriever.run_async(documents=context) if use_async else retriever.run(documents=context)

    assert result == {"documents": context}
