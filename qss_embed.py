#!/usr/bin/env python3
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from qss_common import EMBED_DIM, QSS_WORK, STAGED_INPUT, check_budget, log, reset_output, write_run

INPUT = STAGED_INPUT
TITLE_OUT = QSS_WORK / "embeddings_title"
ABSTRACT_OUT = QSS_WORK / "embeddings_title_abstract"
BATCH_TITLE = 512
BATCH_ABSTRACT = 128
BASE_REVISION = "3447645e1def9117997203454fa4495937bfbd83"
ADAPTER_REVISION = "2081559630a80fc5851d8f798a05ba81e9468089"


def abstract_text(value):
    inverted = json.loads(value)
    if not isinstance(inverted, dict):
        raise ValueError(f"expected abstract JSON object, got {type(inverted).__name__}")
    if not inverted:
        return None, 0
    words = {}
    for word, positions in inverted.items():
        for position in positions:
            if not isinstance(position, int) or position < 0:
                raise ValueError(f"invalid abstract token position {position!r}")
            if position in words:
                raise ValueError(f"duplicate abstract token position {position}")
            words[position] = word
    if not words:
        raise ValueError("abstract inverted index contains no token positions")
    missing_positions = max(words) + 1 - len(words)
    return " ".join(words[position] for position in sorted(words)), missing_positions


def arrow_table(ids, embeddings):
    flat = pa.array(embeddings.reshape(-1), type=pa.float16())
    vectors = pa.FixedSizeListArray.from_arrays(flat, EMBED_DIM)
    return pa.table({"id": pa.array(ids, type=pa.string()), "embedding": vectors})


def encode(model, tokenizer, texts, batch_size, device):
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=512, return_tensors="pt", return_token_type_ids=False,
            ).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model(**tokens).last_hidden_state[:, 0, :]
            blocks.append(functional.normalize(values.float(), dim=1).half().cpu().numpy())
    return __import__("numpy").concatenate(blocks)


def main():
    if not INPUT.is_dir():
        raise FileNotFoundError(f"expected prepared embedding input at {INPUT}")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        reset_output(TITLE_OUT)
        reset_output(ABSTRACT_OUT)
        TITLE_OUT.mkdir(parents=True)
        ABSTRACT_OUT.mkdir(parents=True)
    dist.barrier()
    files = sorted(INPUT.glob("*.parquet"))
    assigned = files[rank::world]
    if not assigned:
        raise ValueError(f"expected input shards for rank {rank}/{world}, got 0")

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", revision=BASE_REVISION)
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base", revision=BASE_REVISION)
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True,
                       revision=ADAPTER_REVISION)
    if list(model.active_adapters.flatten()) != ["specter2"]:
        raise RuntimeError(f"expected active SPECTER2 adapter, got {model.active_adapters}")
    model.eval().to(device)
    title_writer = pq.ParquetWriter(
        TITLE_OUT / f"rank-{rank:02d}.parquet",
        pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.float16(), EMBED_DIM))]),
        compression="zstd",
    )
    abstract_writer = pq.ParquetWriter(
        ABSTRACT_OUT / f"rank-{rank:02d}.parquet",
        pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.float16(), EMBED_DIM))]),
        compression="zstd",
    )
    title_n = abstract_n = empty_abstracts = gap_rows = gap_positions = 0
    for batch in ds.dataset([str(p) for p in assigned], format="parquet").to_batches(
        columns=["id", "title", "abstract_inverted_index", "is_history"], batch_size=4096
    ):
        rows = batch.to_pylist()
        ids = [row["id"] for row in rows]
        titles = [row["title"] for row in rows]
        if any(not title for title in titles):
            raise ValueError("embedding input contains an empty title")
        title_writer.write_table(arrow_table(ids, encode(model, tokenizer, titles, BATCH_TITLE, device)))
        title_n += len(rows)
        selected = [row for row in rows if row["is_history"] and row["abstract_inverted_index"]]
        if selected:
            abstracts = [abstract_text(row["abstract_inverted_index"]) for row in selected]
            usable = [(row, abstract) for row, abstract in zip(selected, abstracts)
                      if abstract[0] is not None]
            empty_abstracts += len(selected) - len(usable)
            texts = [row["title"] + tokenizer.sep_token + abstract[0]
                     for row, abstract in usable]
            gap_rows += sum(abstract[1] > 0 for abstract in abstracts)
            gap_positions += sum(abstract[1] for abstract in abstracts)
            if usable:
                abstract_writer.write_table(arrow_table(
                    [row["id"] for row, _ in usable],
                    encode(model, tokenizer, texts, BATCH_ABSTRACT, device),
                ))
                abstract_n += len(usable)
        if (title_n + abstract_n) % 100_000 < 4096:
            log(f"rank={rank} title={title_n:,} title_abstract={abstract_n:,}")
    title_writer.close()
    abstract_writer.close()
    log(f"rank={rank} complete title={title_n:,} title_abstract={abstract_n:,} "
        f"empty_abstracts={empty_abstracts:,} abstract_gap_rows={gap_rows:,} "
        f"abstract_gap_positions={gap_positions:,}")
    dist.barrier()
    gap_counts = torch.tensor(
        [empty_abstracts, gap_rows, gap_positions], dtype=torch.long, device=device,
    )
    dist.all_reduce(gap_counts, op=dist.ReduceOp.SUM)
    if rank == 0:
        import duckdb
        con = duckdb.connect()
        title_qc = con.execute(
            "SELECT count(*), count(DISTINCT id), min(array_length(embedding)), "
            "max(array_length(embedding)), min(list_inner_product(embedding,embedding)), "
            "max(list_inner_product(embedding,embedding)) FROM read_parquet(?)",
            [str(TITLE_OUT / "*.parquet")],
        ).fetchone()
        abstract_qc = con.execute(
            "SELECT count(*), count(DISTINCT id), min(array_length(embedding)), "
            "max(array_length(embedding)) FROM read_parquet(?)",
            [str(ABSTRACT_OUT / "*.parquet")],
        ).fetchone()
        if title_qc[0] != title_qc[1] or title_qc[2:4] != (EMBED_DIM, EMBED_DIM):
            raise ValueError(f"title embedding QC failed: {title_qc}")
        if not (0.98 <= title_qc[4] <= title_qc[5] <= 1.02):
            raise ValueError(f"title embedding normalization failed: {title_qc[4:6]}")
        if abstract_qc[0] != abstract_qc[1] or abstract_qc[2:4] != (EMBED_DIM, EMBED_DIM):
            raise ValueError(f"title+abstract embedding QC failed: {abstract_qc}")
        write_run("embed", "complete", {
            "title_embeddings": title_qc[0], "title_abstract_embeddings": abstract_qc[0],
            "empty_abstract_objects": int(gap_counts[0].item()),
            "abstract_gap_rows": int(gap_counts[1].item()),
            "abstract_gap_positions": int(gap_counts[2].item()),
        }, {"model": "allenai/specter2_base+allenai/specter2",
            "base_commit": BASE_REVISION, "adapter_commit": ADAPTER_REVISION,
            "world_size": world})
        check_budget()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
