#!/usr/bin/env python3
import hashlib
import html
import json
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

from qss_common import (
    ARTIFACTS, EMBED_DIM, QSS_WORK, SEED, STAGED_INPUT, check_budget, connect,
    log, reset_output, tree_bytes, validate_snapshot, write_run,
)

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RAW = QSS_WORK / "qwen3_embeddings"
SEMANTICS = QSS_WORK / "qwen3_semantics.parquet"
TAXONOMY = ARTIFACTS / "qwen3_taxonomy.npz"
TRAIN_N = 1_000_000
HELDOUT_N = 250_000
LEAVES = 1_000
MACROS = 32
BATCH = 512
MAX_TOKENS = 128
PC_COLS = [f"qpc{i:02d}" for i in range(1, 33)]


def normalized_title(value):
    if not isinstance(value, str):
        raise ValueError(f"expected title string, got {type(value).__name__}")
    value = " ".join(unicodedata.normalize("NFKC", html.unescape(value)).split())
    if not value:
        raise ValueError("title became empty after normalization")
    return value


def work_hash(work_id):
    value = hashlib.blake2b(f"{SEED}|{work_id}".encode(), digest_size=8).digest()
    return int.from_bytes(value, "big")


def vector_array(values):
    flat = pa.array(values.reshape(-1), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(flat, EMBED_DIM)


def vectors(column):
    values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    return np.asarray(values.values, dtype=np.float32).reshape(-1, EMBED_DIM)


def encode(model, tokenizer, titles, device):
    tokens = tokenizer(
        titles, padding=True, truncation=True, max_length=MAX_TOKENS,
        return_tensors="pt", return_token_type_ids=False,
    ).to(device)
    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state[:, -1, :EMBED_DIM]
    return functional.normalize(hidden.float(), dim=1).half().cpu().numpy()


def embed(rank, world, device):
    if rank == 0:
        reset_output(RAW)
        RAW.mkdir(parents=True)
    dist.barrier()
    files = sorted(STAGED_INPUT.glob("*.parquet"))[rank::world]
    if not files:
        raise ValueError(f"expected staged shards for rank {rank}/{world}, got 0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    model = AutoModel.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    got_revision = getattr(model.config, "_commit_hash", None)
    if got_revision != REVISION:
        raise RuntimeError(f"expected Qwen3 revision {REVISION}, got {got_revision}")
    model.eval().to(device)
    schema = pa.schema([
        ("id", pa.string()), ("sample_hash", pa.uint64()),
        ("embedding", pa.list_(pa.float16(), EMBED_DIM)),
    ])
    writer = pq.ParquetWriter(RAW / f"rank-{rank:02d}.parquet", schema, compression="zstd")
    count = 0
    for batch in ds.dataset([str(path) for path in files], format="parquet").to_batches(
        columns=["id", "title"], batch_size=4096,
    ):
        rows = batch.to_pylist()
        ids = [row["id"] for row in rows]
        embeddings = []
        titles = [normalized_title(row["title"]) for row in rows]
        for start in range(0, len(titles), BATCH):
            embeddings.append(encode(model, tokenizer, titles[start:start + BATCH], device))
        matrix = np.concatenate(embeddings)
        writer.write_table(pa.table({
            "id": pa.array(ids),
            "sample_hash": pa.array([work_hash(work_id) for work_id in ids], type=pa.uint64()),
            "embedding": vector_array(matrix),
        }, schema=schema))
        count += len(rows)
        if count % 1_000_000 < 4096:
            log(f"Qwen3 rank={rank} rows={count:,}")
    writer.close()
    log(f"Qwen3 rank={rank} complete rows={count:,}")


def training_sample(con):
    table = con.execute(f"""
        SELECT e.id,e.embedding FROM read_parquet('{RAW}/*.parquet') e
        JOIN read_parquet('{STAGED_INPUT}/*.parquet') s USING (id)
        WHERE s.is_history AND s.history_publication_year BETWEEN 2012 AND 2014
        ORDER BY e.sample_hash,e.id LIMIT {TRAIN_N + HELDOUT_N}
    """).fetch_arrow_table()
    if table.num_rows != TRAIN_N + HELDOUT_N:
        raise ValueError(f"expected {TRAIN_N + HELDOUT_N} taxonomy rows, got {table.num_rows}")
    return vectors(table["embedding"])


def fit_taxonomy(sample):
    train, heldout = sample[:TRAIN_N], sample[TRAIN_N:]
    pca = PCA(n_components=32, svd_solver="randomized", random_state=SEED).fit(train)
    leaf_model = MiniBatchKMeans(
        n_clusters=LEAVES, batch_size=8192, max_iter=100, n_init=3, random_state=SEED,
    ).fit(train)
    train_leaf = leaf_model.predict(train)
    leaf_counts = np.bincount(train_leaf, minlength=LEAVES)
    if np.any(leaf_counts == 0):
        raise ValueError(f"expected {LEAVES} populated leaves, got {(leaf_counts > 0).sum()}")
    macro_model = MiniBatchKMeans(
        n_clusters=MACROS, batch_size=LEAVES, max_iter=100, n_init=10, random_state=SEED,
    ).fit(leaf_model.cluster_centers_, sample_weight=leaf_counts)
    leaf_to_macro = macro_model.predict(leaf_model.cluster_centers_).astype(np.int8)
    heldout_leaf = leaf_model.predict(heldout)
    heldout_macro = leaf_to_macro[heldout_leaf]
    heldout_distance = np.sum(
        (heldout - leaf_model.cluster_centers_[heldout_leaf]) ** 2, axis=1,
    )
    cutoffs = np.empty(MACROS, dtype=np.float32)
    for macro in range(MACROS):
        values = heldout_distance[heldout_macro == macro]
        if len(values) < 100:
            raise ValueError(f"expected >=100 held-out rows in macro {macro}, got {len(values)}")
        cutoffs[macro] = np.quantile(values, 0.99)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TAXONOMY, pca_components=pca.components_, pca_mean=pca.mean_,
        leaf_centers=leaf_model.cluster_centers_, leaf_to_macro=leaf_to_macro,
        macro_centers=macro_model.cluster_centers_, ood_cutoffs=cutoffs,
    )
    return pca, leaf_model, leaf_to_macro, cutoffs


def classify(con, pca, leaf_model, leaf_to_macro, cutoffs):
    reset_output(SEMANTICS)
    schema = pa.schema([
        ("id", pa.string()), ("qwen_leaf", pa.int16()), ("qwen_macro", pa.int8()),
        ("qwen_ood", pa.bool_()), *[(name, pa.float32()) for name in PC_COLS],
    ])
    writer = pq.ParquetWriter(SEMANTICS, schema, compression="zstd")
    reader = con.execute(f"SELECT id,embedding FROM read_parquet('{RAW}/*.parquet')").fetch_record_batch(100_000)
    count = ood_n = 0
    for batch in reader:
        matrix = vectors(batch.column("embedding"))
        leaf = leaf_model.predict(matrix)
        macro = leaf_to_macro[leaf]
        distance = np.sum((matrix - leaf_model.cluster_centers_[leaf]) ** 2, axis=1)
        ood = distance > cutoffs[macro]
        pcs = pca.transform(matrix).astype(np.float32)
        data = {
            "id": batch.column("id"), "qwen_leaf": pa.array(leaf, type=pa.int16()),
            "qwen_macro": pa.array(macro, type=pa.int8()), "qwen_ood": pa.array(ood),
        }
        data.update({name: pa.array(pcs[:, i]) for i, name in enumerate(PC_COLS)})
        writer.write_table(pa.table(data, schema=schema))
        count += len(matrix)
        ood_n += int(ood.sum())
        if count % 1_000_000 < 100_000:
            log(f"Qwen3 classification rows={count:,} OOD={ood_n:,}")
    writer.close()
    return count, ood_n


def main():
    validate_snapshot()
    check_budget()
    if not STAGED_INPUT.is_dir():
        raise FileNotFoundError(f"expected staged input at {STAGED_INPUT}")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    embed(rank, world, device)
    dist.barrier()
    dist.destroy_process_group()
    if rank != 0:
        return
    con = connect("220GB", 32)
    expected = con.execute(f"SELECT count(*) FROM read_parquet('{STAGED_INPUT}/*.parquet')").fetchone()[0]
    qc = con.execute(f"""
        SELECT count(*),count(DISTINCT id),min(list_count(embedding)),max(list_count(embedding)),
               min(list_inner_product(embedding,embedding)),max(list_inner_product(embedding,embedding))
        FROM read_parquet('{RAW}/*.parquet')
    """).fetchone()
    if qc[:4] != (expected, expected, EMBED_DIM, EMBED_DIM):
        raise ValueError(f"Qwen3 embedding QC failed: expected={expected}, got={qc}")
    if not (0.98 <= qc[4] <= qc[5] <= 1.02):
        raise ValueError(f"Qwen3 norm QC failed: {qc[4:6]}")
    raw_bytes = tree_bytes(RAW)
    pca, leaf_model, leaf_to_macro, cutoffs = fit_taxonomy(training_sample(con))
    semantics_n, ood_n = classify(con, pca, leaf_model, leaf_to_macro, cutoffs)
    if semantics_n != expected:
        raise ValueError(f"expected Qwen3 semantics rows={expected}, got {semantics_n}")
    reset_output(RAW)
    write_run("outcome_embed", "complete", {
        "embeddings": expected, "taxonomy_train": TRAIN_N,
        "taxonomy_heldout": HELDOUT_N, "qwen3_semantics": semantics_n, "ood": ood_n,
    }, {
        "model": MODEL, "model_commit": REVISION, "embedding_dimension": EMBED_DIM,
        "raw_embedding_bytes_before_cleanup": raw_bytes, "leaf_clusters": LEAVES,
        "macroclusters": MACROS, "ood_quantile": 0.99,
        "sample_hash": "blake2b-64 big-endian of 20260902|OpenAlex work ID",
        "input": "normalized English title; no prompt", "max_tokens": MAX_TOKENS,
    })
    check_budget()
    log(f"outcome embedding complete rows={semantics_n:,} OOD={ood_n:,}")


if __name__ == "__main__":
    main()
