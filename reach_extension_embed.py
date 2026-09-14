#!/usr/bin/env python3
"""Label only newly required citing titles with the frozen Qwen taxonomy."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from qss_v3_embed import normalized_title
from reach_extension_tenyear import (
    BASE, QWEN_MODEL, QWEN_REVISION, SEED, TAXONOMY, budget, connect,
    log, new_target, prior_manifest, write_manifest,
)


def main():
    prepare = prior_manifest("prepare")
    budget()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(SEED)
    device = torch.device(f"cuda:{rank}")
    target = BASE / "qwen3_semantics"
    pending = target.with_name(target.name + ".partial")
    if rank == 0:
        pending = new_target(target)
        pending.mkdir()
    dist.barrier()
    source = pq.ParquetFile(BASE / "qwen_missing.parquet")
    groups = list(range(rank, source.num_row_groups, world))
    schema = pa.schema([("id", pa.string()), ("qwen_leaf", pa.int16()),
                        ("qwen_macro", pa.int8()), ("qwen_ood", pa.bool_())])
    count, norm_min, norm_max = 0, None, None
    with pq.ParquetWriter(pending / f"rank-{rank:02d}.parquet", schema, compression="zstd") as writer:
        if groups:
            bundle = np.load(TAXONOMY)
            if bundle["leaf_centers"].shape != (1000, 768) or bundle["ood_cutoffs"].shape != (32,):
                raise ValueError("frozen taxonomy dimensions changed")
            centers = torch.tensor(bundle["leaf_centers"], dtype=torch.float32, device=device)
            center_norm = (centers * centers).sum(dim=1)
            macro_map = torch.tensor(bundle["leaf_to_macro"], dtype=torch.long, device=device)
            cutoffs = torch.tensor(bundle["ood_cutoffs"], dtype=torch.float32, device=device)
            tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, revision=QWEN_REVISION, padding_side="left")
            model = AutoModel.from_pretrained(QWEN_MODEL, revision=QWEN_REVISION,
                                              torch_dtype=torch.bfloat16, attn_implementation="sdpa")
            if getattr(model.config, "_commit_hash", None) != QWEN_REVISION:
                raise ValueError(f"expected model revision {QWEN_REVISION}, got {model.config._commit_hash}")
            model.eval().to(device)
            for batch in source.iter_batches(batch_size=512, row_groups=groups, columns=["id", "title"]):
                rows = batch.to_pylist()
                tokens = tokenizer([normalized_title(r["title"]) for r in rows], padding=True,
                                   truncation=True, max_length=128, return_tensors="pt",
                                   return_token_type_ids=False).to(device)
                with torch.inference_mode():
                    vector = functional.normalize(model(**tokens).last_hidden_state[:, -1, :768].float(), dim=1)
                    norms = torch.linalg.vector_norm(vector, dim=1)
                    lo, hi = norms.min().item(), norms.max().item()
                    if vector.shape != (len(rows), 768) or not torch.isfinite(vector).all() or not 0.999 <= lo <= hi <= 1.001:
                        raise ValueError(f"embedding QC failed: shape={vector.shape}, norms=({lo},{hi})")
                    norm_min = lo if norm_min is None else min(norm_min, lo)
                    norm_max = hi if norm_max is None else max(norm_max, hi)
                    distances = (vector * vector).sum(dim=1, keepdim=True) + center_norm[None, :] - 2 * vector @ centers.T
                    leaf = distances.argmin(dim=1)
                    macro = macro_map[leaf]
                    ood = distances.gather(1, leaf[:, None]).squeeze(1) > cutoffs[macro]
                writer.write_table(pa.table({"id": [r["id"] for r in rows],
                    "qwen_leaf": pa.array(leaf.cpu().numpy(), type=pa.int16()),
                    "qwen_macro": pa.array(macro.cpu().numpy(), type=pa.int8()),
                    "qwen_ood": pa.array(ood.cpu().numpy())}, schema=schema))
                count += len(rows)
                if count % 100_000 < 512:
                    budget()
                    log(f"tenyear Qwen rank={rank}: rows={count:,}")
    rank_qc = [None] * world
    dist.all_gather_object(rank_qc, {"rank": rank, "rows": count, "norm_min": norm_min, "norm_max": norm_max})
    dist.barrier()
    if rank == 0:
        con = connect("32GB", 4)
        qc = con.execute(f"SELECT count(*),count(DISTINCT id) FROM read_parquet('{pending}/*.parquet')").fetchone()
        expected = prepare["counts"]["missing_qwen"]
        unmatched = con.execute(f"SELECT count(*) FROM read_parquet('{pending}/*.parquet') a "
            f"FULL JOIN read_parquet('{BASE}/qwen_missing.parquet') b USING (id) WHERE a.id IS NULL OR b.id IS NULL").fetchone()[0]
        if qc != (expected, expected) or unmatched:
            raise ValueError(f"new Qwen labels mismatch: expected={expected}, got={qc}, unmatched={unmatched}")
        budget()
        pending.rename(target)
        write_manifest("embed", {"labels": qc[0]}, {"model": QWEN_MODEL, "revision": QWEN_REVISION,
            "dimensions": 768, "max_tokens": 128, "pooling": "last token, first 768 dimensions, L2 normalized",
            "input": "normalized English titles; no prompt", "rank_qc": rank_qc,
            "raw_embeddings_persisted": False, "taxonomy_retrained": False})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
