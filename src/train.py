import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Paths, TrainConfig, LossConfig
from data import RVLCDIPOCRTextDataset, set_seed
from sampler import MinPerClassBatchSampler
from model import BertDocClassifier
from loss import CustomMarginContrastiveLoss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    paths = Paths(
        project_root=project_root,
        qs_ocr_large_dir=project_root / "QS-OCR-Large",
        rvl_cdip_dir=project_root / "rvl-cdip",
        rvl_cdip_ood_text_dir=project_root / "rvl-cdip-o-text",
        train_list=project_root / "train.txt",
        val_list=project_root / "val.txt",
        test_list=project_root / "test.txt",
    )

    cfg = TrainConfig()
    loss_cfg = LossConfig()

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Load a tiny subset for CPU sanity check.
    # IMPORTANT: sampler needs >= min_per_class per class.
    # If debug_samples=100 makes too many rare classes, the sampler can fail.
    # We solve this by restricting to labels that appear >= min_per_class in this debug subset.
    tmp_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.train_list,
        tokenizer_name=cfg.model_name,
        max_length=cfg.max_length,
        debug_samples=cfg.debug_samples,
    )
    labels_all = [int(tmp_ds.items[i][1]) for i in range(len(tmp_ds))]

    from collections import Counter
    cnt = Counter(labels_all)
    allowed = {lab for lab, c in cnt.items() if c >= cfg.min_per_class}
    if len(allowed) == 0:
        raise RuntimeError(
            "Debug subset has no label with >= min_per_class samples. "
            "Increase debug_samples or reduce min_per_class for the debug run."
        )

    train_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.train_list,
        tokenizer_name=cfg.model_name,
        max_length=cfg.max_length,
        debug_samples=cfg.debug_samples,
        allowed_labels=allowed,
    )

    labels = [int(y) for (_, y) in train_ds.items]
    num_classes = len(set(labels))

    # Map original labels to [0..C-1] because RVL labels might be 0..15 but could differ
    label_to_new = {lab: i for i, lab in enumerate(sorted(set(labels)))}
    # Patch dataset labels in-place (simple, deterministic)
    train_ds.items = [(p, label_to_new[y]) for (p, y) in train_ds.items]
    labels = [label_to_new[y] for y in labels]

    batch_sampler = MinPerClassBatchSampler(labels, cfg.batch_size, cfg.min_per_class, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=cfg.num_workers)

    print("Train dataset size:", len(train_ds))
    print("Num unique labels:", len(set([y for _, y in train_ds.items])))
    print("Train loader len (num batches):", len(train_loader))

    model = BertDocClassifier(cfg.model_name, num_classes=num_classes).to(device)
    criterion = CustomMarginContrastiveLoss(
        alpha=loss_cfg.alpha, beta=loss_cfg.beta, lam=loss_cfg.lam, eps=loss_cfg.eps
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{cfg.epochs}")

        for step, batch in enumerate(pbar):
            if step == 0:
                print(">>> entered training loop, got first batch")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels_t = batch["labels"].to(device)

            optim.zero_grad(set_to_none=True)
            logits, h = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, h, labels_t)
            loss.backward()
            optim.step()

            pbar.set_postfix({"loss": float(loss.detach().cpu())})


    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "bert_margin_star_debug.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_to_new": label_to_new,
            "num_classes": num_classes,
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()