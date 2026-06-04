# -*- coding: utf-8 -*-
"""Управление расчётом task1 на удалённом кластере (ui4-el7.computing.kiae.ru).

Подкоманды:
    submit          — синхронизирует src/*.py и tools/relax.slurm на сервер,
                      sbatch, печатает JOBID
    status [JOBID]  — squeue + sacct (по умолчанию JOBID из tools/.last_jobid)
    fetch  [JOBID]  — scp mixture.out/.skip, f_mixture.npz → data/
    log    [JOBID]  — последние строки output_<JOBID>.out

Подключение через ~/.ssh/config — алиас `kiae`.

Отличие от task0: одна задача (не array), результаты остаются в рабочей
папке ~/task1 (а не переносятся в ~/data).

Использование:
    python tools/cluster.py submit
    python tools/cluster.py status
    python tools/cluster.py fetch
"""
import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer") and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# === Пути ==================================================================
TASK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = TASK_ROOT / "src"
TOOLS_DIR = TASK_ROOT / "tools"
DATA_DIR = TASK_ROOT / "data"
LAST_JOB_FILE = TOOLS_DIR / ".last_jobid"

# === Кластер ===============================================================
HOST = "kiae"                  # из ~/.ssh/config
REMOTE_DIR = "task1"           # рабочий каталог: submit, sbatch, логи, результаты
REMOTE_FETCH_DIR = "task1"     # результаты остаются в рабочей папке
SLURM_FILE = "relax.slurm"

# Файлы кода (src/) + SLURM-скрипт (tools/), которые синхронизируем.
SYNC_PY_FILES = ["relax.py", "sem1.py", "sem2.py", "sem3.py"]

# Файлы результатов, которые скачиваем после завершения.
RESULT_FILES = ["mixture.out", "mixture.skip", "f_mixture.npz"]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", **kw)


def ssh(remote_cmd):
    return run(["ssh", HOST, remote_cmd])


def scp_to(local, remote):
    return run(["scp", "-q", str(local), f"{HOST}:{remote}"])


def scp_from(remote, local):
    return run(["scp", "-q", f"{HOST}:{remote}", str(local)])


def save_jobid(jobid):
    LAST_JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_JOB_FILE.write_text(str(jobid).strip())


def load_jobid():
    if not LAST_JOB_FILE.exists():
        return None
    return LAST_JOB_FILE.read_text().strip() or None


# === submit ================================================================
def cmd_submit(_args):
    uploads = [(SRC_DIR / fn, fn) for fn in SYNC_PY_FILES]
    uploads.append((TOOLS_DIR / SLURM_FILE, SLURM_FILE))

    print(f"==> Синхронизация {len(uploads)} файлов в {HOST}:{REMOTE_DIR}/")
    for local, remote_name in uploads:
        if not local.exists():
            print(f"  [SKIP] нет локально: {local}")
            continue
        r = scp_to(local, f"{REMOTE_DIR}/{remote_name}")
        if r.returncode != 0:
            print(f"  [FAIL] {remote_name}: {r.stderr.strip()}")
            return 1
        print(f"  [OK]   {remote_name}")

    print(f"\n==> sbatch {SLURM_FILE} на сервере")
    r = ssh(f"cd {REMOTE_DIR} && sbatch {SLURM_FILE}")
    if r.returncode != 0:
        print(f"  [FAIL] sbatch: {r.stderr.strip()}")
        return 1
    print(f"  {r.stdout.strip()}")

    m = re.search(r"Submitted batch job (\d+)", r.stdout)
    if not m:
        print("  [WARN] не удалось распарсить JOBID")
        return 1
    jobid = m.group(1)
    save_jobid(jobid)
    print(f"\n  JOBID = {jobid}  (сохранён в tools/.last_jobid)")
    print(f"  Дальше: python tools/cluster.py status  (или log / fetch)")
    return 0


# === status ================================================================
def cmd_status(args):
    jobid = args.jobid or load_jobid()
    if not jobid:
        print("Нет JOBID. Передай аргументом или сначала submit.")
        return 1

    print(f"==> squeue (jobid={jobid})")
    r = ssh(f'squeue -j {jobid} --format="%.10i %.8j %.10T %.10M %.6D %R" 2>&1 || true')
    print(r.stdout.strip() or "(не в очереди)")

    print(f"\n==> sacct (jobid={jobid})")
    r = ssh(f'sacct -j {jobid} --format=JobID,JobName,Elapsed,State,ExitCode --noheader 2>&1 || true')
    print(r.stdout.strip() or "(нет данных в sacct)")
    return 0


# === log ===================================================================
def cmd_log(args):
    jobid = args.jobid or load_jobid()
    if not jobid:
        print("Нет JOBID.")
        return 1
    n = args.lines
    print(f"==> tail -{n} output_{jobid}.out")
    r = ssh(f"tail -{n} {REMOTE_DIR}/output_{jobid}.out 2>/dev/null || echo '(нет файла)'")
    print(r.stdout.rstrip())
    print(f"\n==> tail -{n} output_{jobid}.err")
    r = ssh(f"tail -{n} {REMOTE_DIR}/output_{jobid}.err 2>/dev/null || echo '(нет файла)'")
    print(r.stdout.rstrip())
    return 0


# === fetch =================================================================
def cmd_fetch(_args):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"==> Скачиваю {len(RESULT_FILES)} файлов в {DATA_DIR}/ (из ~/{REMOTE_FETCH_DIR}/)")
    failed = []
    for fn in RESULT_FILES:
        r = scp_from(f"{REMOTE_FETCH_DIR}/{fn}", DATA_DIR / fn)
        if r.returncode == 0:
            print(f"  [OK]   {fn}")
        else:
            print(f"  [FAIL] {fn}: {r.stderr.strip()}")
            failed.append(fn)
    if failed:
        print(f"\n  Не удалось скачать: {failed}")
        return 1
    print(f"\n  Готово. Дальше: запустить data/relax.ipynb, затем python tools/replace_images.py")
    return 0


# === main ==================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("submit", help="залить src/ + relax.slurm на кластер и sbatch")

    p_status = sub.add_parser("status", help="показать состояние задачи")
    p_status.add_argument("jobid", nargs="?", default=None)

    p_log = sub.add_parser("log", help="последние строки output-лога на кластере")
    p_log.add_argument("jobid", nargs="?", default=None)
    p_log.add_argument("-n", "--lines", type=int, default=15)

    sub.add_parser("fetch", help="скачать результаты в data/")

    args = p.parse_args()
    return {
        "submit": cmd_submit,
        "status": cmd_status,
        "log": cmd_log,
        "fetch": cmd_fetch,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
