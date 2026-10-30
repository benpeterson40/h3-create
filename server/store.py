"""store.py : jobs on disk (data/jobs/<id>/job.json) with an in-memory cache; every change is saved."""
import copy, json, os, threading, time, uuid
from . import durable

from .config import DATA   # H3C_DATA overrides (server/config.py)


class Store:
    def __init__(self):
        self.lock = threading.RLock(); self.jobs = {}
        os.makedirs(f"{DATA}/jobs", exist_ok=True)
        for jid in os.listdir(f"{DATA}/jobs"):
            p = f"{DATA}/jobs/{jid}/job.json"
            if not any(os.path.exists(p + x) for x in ("", ".bak", ".tmp")):
                continue
            try:
                self.jobs[jid] = json.load(open(p))
            except Exception:   # unreadable (e.g. a power cut while it was written by a script): its last good version
                j = durable.read_json(p)
                if isinstance(j, dict) and j.get("id"):
                    self.jobs[jid] = j
                    try:
                        durable.write_json(p, j)
                    except OSError:
                        pass
                with open(f"{DATA}/store_recovery.log", "a") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + f"{jid}: job.json unreadable - " +
                            ("restored from its backup\n" if jid in self.jobs else "no readable backup, job skipped\n"))

    def dir(self, jid):
        return f"{DATA}/jobs/{jid}"

    def _save(self, jid):
        d = self.jobs[jid]; os.makedirs(self.dir(jid), exist_ok=True)
        durable.write_json(f"{self.dir(jid)}/job.json", d, keep_bak=True)   # fsynced + atomic: survives a power cut (ratings, comments)

    def create(self, fields):
        with self.lock:
            jid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            self.jobs[jid] = {"id": jid, "created": time.time(), "status": "queued", "stage": "Waiting in line", "progress": 0.0, "log": [], **fields}
            self._save(jid); return jid

    def get(self, jid):
        with self.lock:
            return copy.deepcopy(self.jobs.get(jid))

    def all(self):
        with self.lock:
            return sorted((copy.deepcopy(j) for j in self.jobs.values()), key=lambda j: -j["created"])

    def update(self, jid, **kw):
        with self.lock:
            self.jobs[jid].update(kw); self._save(jid)

    def log(self, jid, msg):
        with self.lock:
            self.jobs[jid].setdefault("log", []).append(time.strftime("%H:%M:%S ") + msg); self._save(jid)

    def add_attempt(self, jid, k, rec):
        with self.lock:
            for s in self.jobs[jid].get("segments", []):
                if s["k"] == k:
                    s["attempts"].append(rec); s["status"] = "checked" if rec.get("audit") else "rendered"
            self._save(jid)

    def seg_done(self, jid, k, rec):
        with self.lock:
            for s in self.jobs[jid].get("segments", []):
                if s["k"] == k:
                    s["status"] = "done"; s["chosen"] = rec["attempt"]
            self._save(jid)

    def remove(self, jid):
        """move a job's folder to data/trash (recoverable) and forget it"""
        with self.lock:
            os.makedirs(f"{DATA}/trash", exist_ok=True)
            if os.path.isdir(self.dir(jid)):
                os.replace(self.dir(jid), f"{DATA}/trash/{jid}")
            self.jobs.pop(jid, None)
