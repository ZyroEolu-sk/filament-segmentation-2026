"""#8: --lr-trace reproduce las bajadas de LR plateau de un run validado sin mirar val."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from train import lr_factor, plateau_trace, traced_mult  # noqa: E402


def test_trace_reproduces_logged_lr(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("step,loss,pq,dice,tp,fp,fn,lr,secs\n"
                   "1500,0.5,0.1,0.5,1,1,1,3.00e-04,10\n"
                   "3000,0.4,0.2,0.5,1,1,1,1.50e-04,20\n"
                   "4500,0.4,0.2,0.5,1,1,1,1.87e-05,30\n\x00\x00")
    trace = plateau_trace(log, lr=3e-4, factor=0.5)
    assert trace == [(1500, 1.0), (3000, 0.5), (4500, 0.0625)]
    assert traced_mult(1500, trace) == 1.0
    assert traced_mult(1501, trace) == 0.5
    assert traced_mult(4500, trace) == 0.0625
    assert traced_mult(9000, trace) == 0.0625  # más allá del log: último multiplicador
    cfg = {"schedule": "plateau", "warmup": 300, "steps": 4500}
    assert lr_factor(150, cfg, traced_mult(150, trace)) == 0.5  # warmup intacto


def test_real_member_trace_matches_log():
    log = ROOT / "runs" / "cnx-s-consensus" / "log.csv"
    if not log.exists():
        return
    trace = plateau_trace(log, lr=3e-4, factor=0.5)
    assert traced_mult(21000, trace) * 3e-4 == 3.75e-05
