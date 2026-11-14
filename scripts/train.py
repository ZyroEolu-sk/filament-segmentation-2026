"""Entrena FilamentNet con el generador de recortes y valida con PQ sobre imágenes completas.

Uso: ~/venvs/ml/bin/python scripts/train.py --name r34 --encoder resnet34 --steps 8000
Salida: runs/<name>/{config.json, log.csv, best.pt, last.pt}
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filaments.device import autocast, empty_cache, grad_scaler, pick  # noqa: E402
from filaments.data import CACHE, CropGenerator  # noqa: E402
from filaments.infer import predict  # noqa: E402
from filaments.model import SegLoss, build  # noqa: E402
from filaments.postprocess import PostCfg, instances, pq, pq_counts  # noqa: E402


def evaluate(model, stems, annotations, tta=False, device=None) -> dict:
    """PQ contra todos los anotadores de cada observación + Dice de píxel."""
    model.eval()
    device = pick(device)
    probs = {s: predict(model, np.load(CACHE / "images" / f"{s}.npy"), tta=tta, device=device) for s in stems}
    model.train()
    empty_cache(device)  # devolver al driver la memoria de la evaluación: en WSL un pico alto acaba paginado a RAM

    def one(s):
        lab = instances(probs[s], PostCfg())
        out, dices = [], []
        for a in annotations[s]:
            gt = np.load(CACHE / "labels" / f"{a}.npy")
            out.append(pq_counts(gt, lab))
            g, p = gt > 0, probs[s][0] > 0.5
            dices.append(2 * (g & p).sum() / max(g.sum() + p.sum(), 1))
        return out, dices

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, stems))
    counts = [c for r in res for c in r[0]]
    return {"pq": pq(counts), "dice": float(np.mean([d for r in res for d in r[1]])),
            "tp": sum(c[1] for c in counts), "fp": sum(c[2] for c in counts), "fn": sum(c[3] for c in counts)}


def lr_factor(step: int, cfg: dict, plateau_mult: float) -> float:
    """Multiplicador del LR base: warmup lineal × esquema elegido.

    cosine   : un único coseno hasta 0 en cfg["steps"].
    restarts : SGDR, coseno que vuelve a LR alto cada ciclo (cfg["cycle"] pasos, ×cfg["cycle_mult"]
               por ciclo); el salto de LR saca al modelo de mínimos estrechos.
    plateau  : LR constante que se multiplica por cfg["lr_factor"] cuando el PQ de validación se estanca.
    """
    warm = min(1.0, step / cfg["warmup"])
    if cfg["schedule"] == "cosine":
        return warm * 0.5 * (1 + np.cos(np.pi * min(step, cfg["steps"]) / cfg["steps"]))
    if cfg["schedule"] == "restarts":
        t, length = step - 1, cfg["cycle"]  # el ciclo acaba en el LR mínimo justo cuando toca evaluar
        while t >= length:
            t -= length
            length = int(length * cfg["cycle_mult"])
        return warm * (cfg["lr_min"] + (1 - cfg["lr_min"]) * 0.5 * (1 + np.cos(np.pi * t / length)))
    return warm * plateau_mult


def plateau_trace(log_csv: Path, lr: float, factor: float) -> list[tuple[int, float]]:
    """#8: multiplicadores plateau registrados en el log.csv de un run validado, como [(último paso, mult), ...].

    El LR anotado en la evaluación del paso s es el que se usó en los pasos (s_anterior, s]; las bajadas solo
    ocurren tras una evaluación. El multiplicador se redondea a una potencia de `factor` porque el log
    guarda el LR con tres cifras (1.87e-05 en vez de 1.875e-05).
    """
    rows = []
    for line in log_csv.read_text().replace("\x00", "").splitlines():
        cols = line.split(",")
        if len(cols) < 8 or not cols[0].isdigit():
            continue
        mult = factor ** round(np.log(float(cols[7]) / lr) / np.log(factor))
        rows.append((int(cols[0]), float(mult)))
    if not rows:
        raise ValueError(f"{log_csv}: sin filas de evaluación con LR")
    return rows


def traced_mult(step: int, trace: list[tuple[int, float]]) -> float:
    for last, mult in trace:
        if step <= last:
            return mult
    return trace[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init-checkpoint", default="", help="opt-in: iniciar pesos desde un checkpoint previo")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--augmentation", choices=["standard", "solar"], default="standard")
    ap.add_argument("--inference-tile", type=int, default=0, help="opt-in: tiled inference for transformer memory")
    ap.add_argument("--inference-overlap", type=int, default=128)
    ap.add_argument("--no-maxpool", action="store_true", help="opt-in ResNet: conservar resolución ×2 en las capas profundas")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--max-seconds", type=float, default=0, help="opt-in wall-clock budget; evaluate and save before stopping")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--full", action="store_true", help="entrenar con train+val (para el envío final)")
    ap.add_argument("--seed", type=int, default=0)
    # objetivo (ideas #6/#18 y #19; por defecto, el comportamiento original)
    ap.add_argument("--target", choices=["random", "consensus", "majority", "consensus-ignore"], default="random",
                    help="random: un anotador al azar por recorte; consensus: fracción de anotadores (blando); majority: >= mitad; "
                         "consensus-ignore: consensus sin loss en píxeles que marca solo una minoría")
    ap.add_argument("--hard-neg", default="", help="JSON {stem: [[y, x], ...]} de falsos positivos difíciles (scripts/hard_negatives.py)")
    ap.add_argument("--p-hard", type=float, default=0.2, help="fracción de recortes centrados en negativos difíciles (si --hard-neg)")
    ap.add_argument("--p-focus", type=float, default=0.8,
                    help="probabilidad de centrar en filamento condicionada a no elegir negativo difícil")
    ap.add_argument("--year-weight", action="append", default=[], metavar="AÑO:PESO",
                    help="#7: peso relativo de las observaciones de ese año al muestrear (repetible, p.ej. 2021:2); por defecto uniforme")
    # loss
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce", help="término de píxel para el filamento (+ Dice siempre)")
    ap.add_argument("--skel-weight", type=float, default=0.0, help="peso de la Skeleton Recall Loss (0 = off, p.ej. 1.0)")
    # variación del learning rate
    ap.add_argument("--schedule", choices=["cosine", "restarts", "plateau"], default="cosine")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--cycle", type=int, default=2000, help="restarts: pasos del primer ciclo")
    ap.add_argument("--cycle-mult", type=float, default=1.0, help="restarts: factor de longitud entre ciclos")
    ap.add_argument("--lr-min", type=float, default=0.01, help="restarts: LR mínimo del ciclo (fracción del base)")
    ap.add_argument("--lr-patience", type=int, default=2, help="plateau: evaluaciones sin mejora antes de bajar el LR")
    ap.add_argument("--lr-factor", type=float, default=0.5, help="plateau: factor de bajada del LR")
    ap.add_argument("--lr-trace", default="",
                    help="#8 opt-in: log.csv de un run plateau validado; reproduce sus bajadas de LR por paso sin "
                         "mirar val (permite --full con --schedule plateau)")
    # escapar de mínimos locales / mínimos más planos
    ap.add_argument("--ema", type=float, default=0.0, help="decay de la media exponencial de pesos (0 = off, p.ej. 0.999)")
    # early stopping
    ap.add_argument("--patience", type=int, default=0, help="evaluaciones sin mejora antes de parar (0 = off)")
    ap.add_argument("--min-delta", type=float, default=0.002, help="mejora mínima de PQ que cuenta como mejora")
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto (el primero disponible)")
    ap.add_argument("--fold", type=int, default=-1, help="#14: excluir del entrenamiento el fold k de train (0..folds-1); -1 = usar todo train")
    ap.add_argument("--folds", type=int, default=5, help="#14: número de folds por observación sobre train")
    ap.add_argument("--fold-seed", type=int, default=20260926, help="#14: semilla del reparto en folds")
    cfg = vars(ap.parse_args())
    if not 0 <= cfg["p_focus"] <= 1 or not 0 <= cfg["p_hard"] <= 1:
        ap.error("--p-focus y --p-hard deben estar entre 0 y 1")
    if cfg["lr_trace"] and cfg["schedule"] != "plateau":
        ap.error("--lr-trace solo tiene sentido con --schedule plateau")
    if cfg["full"] and ((cfg["schedule"] == "plateau" and not cfg["lr_trace"]) or cfg["patience"]):
        ap.error("plateau (sin --lr-trace) y --patience necesitan validación; no se pueden usar con --full")
    trace = plateau_trace(Path(cfg["lr_trace"]), cfg["lr"], cfg["lr_factor"]) if cfg["lr_trace"] else None

    torch.manual_seed(cfg["seed"])
    device = pick(cfg.pop("device"))
    print(f"dispositivo: {device}", flush=True)
    # velocidad: kernels autoajustados al tamaño fijo del recorte, TF32 y formato NHWC (tensor cores en fp16).
    # No cambian la función que se optimiza; benchmark introduce una ligera no-determinismo entre runs.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = ROOT / "runs" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((CACHE / "index.json").read_text())
    train_stems = index["train"] + (index["val"] if cfg["full"] else [])
    if cfg["fold"] >= 0:
        if cfg["full"]:
            ap.error("--fold y --full son incompatibles")
        from filaments.folds import split_folds
        heldout = split_folds(index["train"], cfg["folds"], cfg["fold_seed"])[cfg["fold"]]
        train_stems = [s for s in index["train"] if s not in set(heldout)]
        (out / "heldout.json").write_text(json.dumps(heldout, indent=1))
        print(f"fold {cfg['fold']}/{cfg['folds']}: {len(train_stems)} observaciones de entrenamiento, {len(heldout)} excluidas", flush=True)

    year_weights = {}
    for item in cfg["year_weight"]:
        year, _, weight = item.partition(":")
        try:
            year_weights[year] = float(weight)
        except ValueError:
            ap.error(f"--year-weight inválido: {item} (formato AÑO:PESO)")
        if len(year) != 4 or not year.isdigit() or year_weights[year] <= 0:
            ap.error(f"--year-weight inválido: {item} (formato AÑO:PESO, peso > 0)")
    hard = json.loads(Path(cfg["hard_neg"]).read_text()) if cfg["hard_neg"] else None
    if cfg["hard_neg"]:
        if not hard:
            ap.error("El archivo de negativos difíciles está vacío")
        snapshot = json.dumps(hard, sort_keys=True, separators=(",", ":"))
        (out / "hard_negatives_snapshot.json").write_text(snapshot + "\n")
        cfg["hard_neg_sha256"] = hashlib.sha256(snapshot.encode()).hexdigest()
        metadata = Path(cfg["hard_neg"]).with_suffix(".meta.json")
        if metadata.exists():
            (out / "hard_negatives_source.meta.json").write_text(metadata.read_text())
    gen = CropGenerator(train_stems, crop=cfg["crop"], seed=cfg["seed"], target=cfg["target"],
                        hard_negatives=hard, p_hard=cfg["p_hard"] if hard else 0.0, p_focus=cfg["p_focus"],
                        year_weights=year_weights or None, augmentation=cfg["augmentation"])
    if year_weights:
        recent = sum(p for s, p in zip(gen.stems, gen.stem_p) if s[:4] in year_weights)
        cfg["year_weight_fraction"] = recent
        print(f"pesos por año {year_weights}: {recent:.3f} de las muestras (sin pesos {sum(s[:4] in year_weights for s in gen.stems) / len(gen.stems):.3f})", flush=True)
    cfg["sampling_fractions"] = {"hard": gen.p_hard, "focus": (1 - gen.p_hard) * gen.p_focus,
                                 "random": (1 - gen.p_hard) * (1 - gen.p_focus)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"fracciones de recortes (nominales, con centroides): {cfg['sampling_fractions']}", flush=True)
    if hard:
        print(f"negativos difíciles: {sum(len(v) for v in gen.hard.values())} puntos en {len(gen.hard)} imágenes; p_hard {gen.p_hard}", flush=True)
    loader = DataLoader(gen, batch_size=cfg["batch"],
                        num_workers=cfg["workers"], pin_memory=True, persistent_workers=True, prefetch_factor=4)
    model = build(cfg).to(device).to(memory_format=torch.channels_last)
    if cfg["init_checkpoint"]:
        model.load_state_dict(torch.load(cfg["init_checkpoint"], map_location=device, weights_only=True))
        print(f"pesos iniciales: {cfg['init_checkpoint']}", flush=True)
    # transfer learning: el encoder preentrenado aprende 10x más despacio que el decoder nuevo
    enc = list(model.net.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg["lr"] / 10}, {"params": rest, "lr": cfg["lr"]}], weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = grad_scaler(device)
    loss_fn = SegLoss(fg_loss=cfg["loss"], skel_weight=cfg["skel_weight"]).to(device)
    # EMA de los pesos (incluye buffers de BatchNorm): suaviza el ruido de SGD y cae en mínimos más planos
    ema = (torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]),
                                               use_buffers=True) if cfg["ema"] else None)

    log = open(out / "log.csv", "a")
    log.write("step,loss,pq,dice,tp,fp,fn,lr,secs\n")
    best, run_loss, t0 = -1.0, 0.0, time.time()
    plateau_mult, since_best, since_lr_drop = 1.0, 0, 0
    it = iter(loader)
    for step in range(1, cfg["steps"] + 1):
        if trace:
            plateau_mult = traced_mult(step, trace)
        factor = lr_factor(step, cfg, plateau_mult)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * factor
        x, y = next(it)
        x, y = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last), y.to(device, non_blocking=True)
        with autocast(device):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            ema.update_parameters(model)
        run_loss += loss.item()

        time_limit = cfg["max_seconds"] > 0 and time.time() - t0 >= cfg["max_seconds"]
        if step % cfg["eval_every"] == 0 or step == cfg["steps"] or time_limit:
            net = ema.module if ema is not None else model  # se evalúa y se guarda la versión EMA
            m = evaluate(net, index["val"], index["annotations"], device=device) if not cfg["full"] else {}
            avg = run_loss / cfg["eval_every"]
            run_loss = 0.0
            secs = time.time() - t0
            lr = opt.param_groups[-1]["lr"]
            log.write(f"{step},{avg:.4f},{m.get('pq', '')},{m.get('dice', '')},{m.get('tp', '')},{m.get('fp', '')},{m.get('fn', '')},{lr:.2e},{secs:.0f}\n")
            log.flush()
            print(f"step {step} loss {avg:.4f} " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()) + f" lr {lr:.2e} [{secs:.0f}s]", flush=True)
            torch.save(net.state_dict(), out / "last.pt")
            if m.get("pq", 0) > best + cfg["min_delta"]:
                since_best = since_lr_drop = 0
            else:
                since_best += 1
                since_lr_drop += 1
            if "pq" in m and m["pq"] > best:  # con --full no hay evaluación: solo last.pt
                best = m["pq"]
                torch.save(net.state_dict(), out / "best.pt")
            if cfg["schedule"] == "plateau" and not trace and since_lr_drop >= cfg["lr_patience"]:
                plateau_mult *= cfg["lr_factor"]
                since_lr_drop = 0
                print(f"  PQ estancado {cfg['lr_patience']} evaluaciones -> LR x{cfg['lr_factor']} (multiplicador {plateau_mult:.3g})", flush=True)
            if cfg["patience"] and since_best >= cfg["patience"]:
                print(f"  early stopping: {since_best} evaluaciones sin mejorar {best:.4f} en más de {cfg['min_delta']}", flush=True)
                break
            if time_limit:
                print("wall-clock budget reached; checkpoint saved", flush=True)
                break
    print(f"mejor PQ val (postproceso por defecto): {best:.4f}")


if __name__ == "__main__":
    main()
