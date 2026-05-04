# -*- coding: utf-8 -*-
"""Управление расчётами на удалённом кластере (ui4-el7.computing.kiae.ru).

Подкоманды:
    submit              — синхронизирует src/*.py на сервер, sbatch, печатает JOBID
    status [JOBID]      — squeue + sacct, показывает Elapsed/State (по умолчанию
                          использует JOBID из tools/.last_jobid)
    fetch  [JOBID]      — scp 05/10/15.{out,skip}, f_*.npz → data/
    log    [JOBID]      — показать последние строки output_<JOBID>_<arrayid>.out
                          для всех трёх array tasks

Подключение настроено через ~/.ssh/config — алиас `kiae`.

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

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# === Пути ==================================================================
TASK_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = TASK_ROOT / "src"
DATA_DIR  = TASK_ROOT / "data"
LAST_JOB_FILE = TASK_ROOT / "tools" / ".last_jobid"

# === Кластер ===============================================================
HOST = "kiae"                              # из ~/.ssh/config
REMOTE_DIR       = "task0"                 # рабочий каталог: submit, sbatch, логи
REMOTE_FETCH_DIR = "data"                  # туда SLURM-скрипт переносит результаты (mv 05.* … ../data/) оттуда забираем
SLURM_FILE = "relax.slurm"                 # имя SLURM-скрипта на кластере (уже там)

# Локальные файлы кода, которые надо синхронизировать перед запуском.
SYNC_FILES = ["relax.py", "sem1.py", "sem2.py", "sem3.py"]

# Файлы результатов, которые скачиваем после завершения.
RESULT_FILES = [
    "05.out", "10.out", "15.out",
    "05.skip", "10.skip", "15.skip",
    "f_05.npz", "f_10.npz", "f_15.npz",
]


def run(cmd, **kw):
    """Запускает команду, возвращает CompletedProcess. Печатает stderr в случае ошибки."""
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", **kw)
    return res


def ssh(remote_cmd):
    """Выполняет команду на удалённой стороне через ssh алиас."""
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
    print(f"==> Синхронизация {len(SYNC_FILES)} файлов в {HOST}:{REMOTE_DIR}/")
    for fn in SYNC_FILES:
        local = SRC_DIR / fn
        if not local.exists():
            print(f"  [SKIP] нет локально: {local}")
            continue
        r = scp_to(local, f"{REMOTE_DIR}/{fn}")
        if r.returncode != 0:
            print(f"  [FAIL] {fn}: {r.stderr.strip()}")
            return 1
        print(f"  [OK]   {fn}")

    # Локального job.slurm у пользователя сейчас нет — используем тот, что уже лежит на сервере как relax.slurm.
    print(f"\n==> sbatch {SLURM_FILE} на сервере")
    r = ssh(f"cd {REMOTE_DIR} && sbatch {SLURM_FILE}")
    if r.returncode != 0:
        print(f"  [FAIL] sbatch: {r.stderr.strip()}")
        return 1
    print(f"  {r.stdout.strip()}")

    m = re.search(r"Submitted batch job (\d+)", r.stdout)
    if not m:
        print("  [WARN] не удалось распарсить JOBID из вывода sbatch")
        return 1
    jobid = m.group(1)
    save_jobid(jobid)
    print(f"\n  JOBID = {jobid}  (сохранён в tools/.last_jobid)")
    print(f"  Дальше: python tools/cluster.py status  (или watch / fetch)")
    return 0


# === status ================================================================
def cmd_status(args):
    jobid = args.jobid or load_jobid()
    if not jobid:
        print("Нет JOBID. Передай аргументом или сначала сделай submit.")
        return 1

    # squeue — для текущей очереди, sacct — для финальных состояний.
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
    for arr in (0, 1, 2):
        u = (0.5, 1.0, 1.5)[arr]
        print(f"\n==> u={u}: tail -{n} output_{jobid}_{arr}.out")
        r = ssh(f"tail -{n} {REMOTE_DIR}/output_{jobid}_{arr}.out 2>/dev/null || echo '(нет файла)'")
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

    sub.add_parser("submit", help="залить src/ на кластер и sbatch")

    p_status = sub.add_parser("status", help="показать состояние задачи")
    p_status.add_argument("jobid", nargs="?", default=None)

    p_log = sub.add_parser("log", help="последние строки output-логов на кластере")
    p_log.add_argument("jobid", nargs="?", default=None)
    p_log.add_argument("-n", "--lines", type=int, default=1)

    sub.add_parser("fetch", help="скачать результаты в data/")

    args = p.parse_args()
    return {
        "submit": cmd_submit,
        "status": cmd_status,
        "log":    cmd_log,
        "fetch":  cmd_fetch,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
