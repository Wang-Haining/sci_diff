#!/usr/bin/env python3
import gc
import html
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from adapters import AutoAdapterModel
from transformers import AutoModel, AutoTokenizer

from qss_common import EMBED_DIM, REPO
from qss_v3_common import (
    V3_WORK, check_budget, connect, log, reset_output, tree_bytes,
    validate_snapshot, write_run,
)

QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
SPECTER_BASE = "3447645e1def9117997203454fa4495937bfbd83"
SPECTER_ADAPTER = "2081559630a80fc5851d8f798a05ba81e9468089"
QWEN_INPUT = V3_WORK / "qwen_input"
SPECTER_INPUT = V3_WORK / "specter_input"
QWEN_OUT = V3_WORK / "qwen3_semantics"
SPECTER_OUT = V3_WORK / "specter_embeddings"
TAXONOMY = REPO / "artifacts/qss_v2/qwen3_taxonomy.npz"


def normalized_title(value):
    if not isinstance(value, str):
        raise ValueError(f"expected title string, got {type(value).__name__}")
    value = " ".join(unicodedata.normalize("NFKC", html.unescape(value)).split())
    if not value:
        raise ValueError("title became empty after normalization")
    return value


def fixed_vectors(values):
    flat = pa.array(values.reshape(-1), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(flat, EMBED_DIM)


def qwen_stage(rank, world, device):
    bundle = np.load(TAXONOMY)
    centers = torch.tensor(bundle["leaf_centers"], dtype=torch.float32, device=device)
    center_norm = (centers * centers).sum(dim=1)
    leaf_to_macro = torch.tensor(bundle["leaf_to_macro"], dtype=torch.long, device=device)
    cutoffs = torch.tensor(bundle["ood_cutoffs"], dtype=torch.float32, device=device)
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL, revision=QWEN_REVISION, padding_side="left",
    )
    model = AutoModel.from_pretrained(
        QWEN_MODEL, revision=QWEN_REVISION, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if getattr(model.config, "_commit_hash", None) != QWEN_REVISION:
        raise RuntimeError(f"expected Qwen3 revision {QWEN_REVISION}, "
                           f"got {getattr(model.config, '_commit_hash', None)}")
    model.eval().to(device)
    files = sorted(QWEN_INPUT.glob("*.parquet"))[rank::world]
    if not files:
        raise ValueError(f"expected Qwen input shards for rank {rank}/{world}, got 0")
    writer = pq.ParquetWriter(
        QWEN_OUT / f"rank-{rank:02d}.parquet",
        pa.schema([("id", pa.string()), ("qwen_leaf", pa.int16()),
                   ("qwen_macro", pa.int8()), ("qwen_ood", pa.bool_())]),
        compression="zstd",
    )
    count = 0
    for batch in ds.dataset([str(path) for path in files], format="parquet").to_batches(
        columns=["id", "title"], batch_size=4096,
    ):
        rows = batch.to_pylist()
        ids = [row["id"] for row in rows]
        titles = [normalized_title(row["title"]) for row in rows]
        leaves, macros, oods = [], [], []
        for start in range(0, len(titles), 512):
            tokens = tokenizer(
                titles[start:start + 512], padding=True, truncation=True,
                max_length=128, return_tensors="pt", return_token_type_ids=False,
            ).to(device)
            with torch.inference_mode():
                values = model(**tokens).last_hidden_state[:, -1, :EMBED_DIM]
                values = functional.normalize(values.float(), dim=1)
                distances = ((values * values).sum(dim=1, keepdim=True)
                             + center_norm[None, :] - 2 * values @ centers.T)
                leaf = distances.argmin(dim=1)
                distance = distances.gather(1, leaf[:, None]).squeeze(1)
                macro = leaf_to_macro[leaf]
                ood = distance > cutoffs[macro]
            leaves.append(leaf.short().cpu().numpy())
            macros.append(macro.byte().cpu().numpy())
            oods.append(ood.cpu().numpy())
        writer.write_table(pa.table({
            "id": pa.array(ids),
            "qwen_leaf": pa.array(np.concatenate(leaves), type=pa.int16()),
            "qwen_macro": pa.array(np.concatenate(macros), type=pa.int8()),
            "qwen_ood": pa.array(np.concatenate(oods)),
        }))
        count += len(rows)
        if count % 1_000_000 < 4096:
            log(f"v3 Qwen3 rank={rank} rows={count:,}")
    writer.close()
    del model, tokenizer, centers, center_norm, leaf_to_macro, cutoffs
    gc.collect()
    torch.cuda.empty_cache()
    log(f"v3 Qwen3 rank={rank} complete rows={count:,}")


def specter_stage(rank, world, device):
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", revision=SPECTER_BASE)
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base", revision=SPECTER_BASE)
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="specter2", set_active=True,
        revision=SPECTER_ADAPTER,
    )
    if list(model.active_adapters.flatten()) != ["specter2"]:
        raise RuntimeError(f"expected active SPECTER2 adapter, got {model.active_adapters}")
    model.eval().to(device)
    files = sorted(SPECTER_INPUT.glob("*.parquet"))[rank::world]
    if not files:
        raise ValueError(f"expected SPECTER input shards for rank {rank}/{world}, got 0")
    writer = pq.ParquetWriter(
        SPECTER_OUT / f"rank-{rank:02d}.parquet",
        pa.schema([("id", pa.string()),
                   ("embedding", pa.list_(pa.float16(), EMBED_DIM))]),
        compression="zstd",
    )
    count = 0
    for batch in ds.dataset([str(path) for path in files], format="parquet").to_batches(
        columns=["id", "title"], batch_size=4096,
    ):
        rows = batch.to_pylist()
        ids = [row["id"] for row in rows]
        titles = [normalized_title(row["title"]) for row in rows]
        blocks = []
        for start in range(0, len(titles), 512):
            tokens = tokenizer(
                titles[start:start + 512], padding=True, truncation=True,
                max_length=512, return_tensors="pt", return_token_type_ids=False,
            ).to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model(**tokens).last_hidden_state[:, 0, :]
            blocks.append(functional.normalize(values.float(), dim=1).half().cpu().numpy())
        writer.write_table(pa.table({
            "id": pa.array(ids), "embedding": fixed_vectors(np.concatenate(blocks)),
        }))
        count += len(rows)
        if count % 1_000_000 < 4096:
            log(f"v3 SPECTER2 rank={rank} rows={count:,}")
    writer.close()
    log(f"v3 SPECTER2 rank={rank} complete rows={count:,}")


def main():
    validate_snapshot()
    if not TAXONOMY.is_file():
        raise FileNotFoundError(f"expected frozen taxonomy at {TAXONOMY}")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        reset_output(QWEN_OUT)
        reset_output(SPECTER_OUT)
        QWEN_OUT.mkdir(parents=True)
        SPECTER_OUT.mkdir(parents=True)
    dist.barrier()
    qwen_stage(rank, world, device)
    dist.barrier()
    specter_stage(rank, world, device)
    dist.barrier()
    if rank == 0:
        con = connect("220GB", 32)
        qwen_expected = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(QWEN_INPUT / "*.parquet")],
        ).fetchone()[0]
        qwen_qc = con.execute(
            "SELECT count(*),count(DISTINCT id) FROM read_parquet(?)",
            [str(QWEN_OUT / "*.parquet")],
        ).fetchone()
        specter_expected = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(SPECTER_INPUT / "*.parquet")],
        ).fetchone()[0]
        specter_qc = con.execute(
            "SELECT count(*),count(DISTINCT id),min(array_length(embedding)),"
            "max(array_length(embedding)),min(list_inner_product(embedding,embedding)),"
            "max(list_inner_product(embedding,embedding)) FROM read_parquet(?)",
            [str(SPECTER_OUT / "*.parquet")],
        ).fetchone()
        if qwen_qc != (qwen_expected, qwen_expected):
            raise ValueError(f"Qwen3 QC failed: expected={qwen_expected}, got={qwen_qc}")
        if specter_qc[:4] != (specter_expected, specter_expected, EMBED_DIM, EMBED_DIM):
            raise ValueError(f"SPECTER2 QC failed: expected={specter_expected}, got={specter_qc}")
        if not 0.98 <= specter_qc[4] <= specter_qc[5] <= 1.02:
            raise ValueError(f"SPECTER2 norm QC failed: {specter_qc[4:6]}")
        write_run("embed", {
            "qwen3_semantics": qwen_qc[0], "specter_embeddings": specter_qc[0],
        }, {
            "qwen3_model_commit": QWEN_REVISION,
            "specter_base_commit": SPECTER_BASE,
            "specter_adapter_commit": SPECTER_ADAPTER,
            "embedding_dimension": EMBED_DIM, "world_size": world,
            "qwen3_bytes": tree_bytes(QWEN_OUT),
            "specter_bytes": tree_bytes(SPECTER_OUT),
        })
        check_budget()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
