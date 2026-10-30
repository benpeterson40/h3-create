"""durable.py : power-cut-safe file writes (user 2026-10-26: "If we loose power let's make sure we don't loose anything").
write_json / write_bytes / copy_file write to <path>.tmp, flush + fsync it, rename it over <path> (atomic on the same filesystem) and fsync the
folder, so a file is either the old version or the new one - never half written - and a saved change survives a power cut right after it.
keep_bak=True also keeps the previous version as <path>.bak (a hard link of the old file, no extra copy). read_json falls back to .bak and .tmp
when <path> is missing or unreadable."""
import json, os, shutil


def _fsync_dir(d):
    try:
        fd = os.open(d or ".", os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _keep_bak(path):
    if os.path.exists(path):
        try:
            if os.path.exists(path + ".bak.tmp"):
                os.remove(path + ".bak.tmp")
            os.link(path, path + ".bak.tmp"); os.replace(path + ".bak.tmp", path + ".bak")
        except OSError:
            pass


def write_bytes(path, data, keep_bak=False):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    if keep_bak:
        _keep_bak(path)
    os.replace(tmp, path); _fsync_dir(os.path.dirname(path))


def write_json(path, obj, indent=1, keep_bak=False):
    write_bytes(path, json.dumps(obj, indent=indent).encode("utf-8"), keep_bak=keep_bak)


def copy_file(src, dst):
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst); _fsync_dir(os.path.dirname(dst))


def read_json(path, default=None):
    """the file, else its last good version (.bak), else an unfinished new version (.tmp); default when none can be read"""
    for p in (path, path + ".bak", path + ".tmp"):
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            continue
    return default
