"""Small, persistent batch queue for the interview demo's single server process."""

import base64
import binascii
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from materials import extract_bundle
from reviewer import review


MAX_BATCH_ITEMS = 100
MAX_PENDING_ITEMS = 100
MAX_DAILY_ITEMS = 200
RETENTION_HOURS = 24


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class BatchStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, client TEXT NOT NULL, campaign TEXT NOT NULL,
                    channel TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches(id),
                    external_id TEXT NOT NULL, position INTEGER NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL,
                    report TEXT, error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    UNIQUE(batch_id, external_id)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, updated_at);
                CREATE INDEX IF NOT EXISTS jobs_batch ON jobs(batch_id, position);
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    batch_id TEXT NOT NULL, digest TEXT NOT NULL, material TEXT NOT NULL,
                    PRIMARY KEY(batch_id,digest)
                );
                CREATE TABLE IF NOT EXISTS source_assets (
                    job_id TEXT NOT NULL, position INTEGER NOT NULL,
                    filename TEXT NOT NULL, mime TEXT NOT NULL, content BLOB NOT NULL,
                    PRIMARY KEY(job_id,position)
                );
                CREATE TABLE IF NOT EXISTS review_decisions (
                    job_id TEXT PRIMARY KEY, decision TEXT NOT NULL, reviewer TEXT NOT NULL,
                    reason TEXT NOT NULL, decided_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    decision TEXT NOT NULL, reviewer TEXT NOT NULL,
                    reason TEXT NOT NULL, decided_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_events_job ON review_events(job_id,id);
            """)
            # A restart cannot know whether a previous model call completed.
            db.execute("UPDATE jobs SET status='queued', error='服务重启，已重新排队' WHERE status='running'")
        self.purge_expired()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def purge_expired(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=RETENTION_HOURS)).isoformat()
        with self.connect() as db:
            db.execute("DELETE FROM source_assets WHERE job_id IN (SELECT jobs.id FROM jobs JOIN batches ON jobs.batch_id=batches.id WHERE batches.created_at < ?)", (cutoff,))
            db.execute("DELETE FROM review_events WHERE job_id IN (SELECT jobs.id FROM jobs JOIN batches ON jobs.batch_id=batches.id WHERE batches.created_at < ?)", (cutoff,))
            db.execute("DELETE FROM review_decisions WHERE job_id IN (SELECT jobs.id FROM jobs JOIN batches ON jobs.batch_id=batches.id WHERE batches.created_at < ?)", (cutoff,))
            db.execute("DELETE FROM ocr_cache WHERE batch_id IN (SELECT id FROM batches WHERE created_at < ?)", (cutoff,))
            db.execute("DELETE FROM jobs WHERE batch_id IN (SELECT id FROM batches WHERE created_at < ?)", (cutoff,))
            db.execute("DELETE FROM batches WHERE created_at < ?", (cutoff,))

    def cached_ocr(self, batch_id, digest):
        with self.connect() as db:
            row = db.execute("SELECT material FROM ocr_cache WHERE batch_id=? AND digest=?", (batch_id, digest)).fetchone()
        return json.loads(row['material']) if row else None

    def save_ocr(self, batch_id, digest, material):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO ocr_cache VALUES (?,?,?)", (batch_id, digest, json.dumps(material, ensure_ascii=False)))

    def create(self, client, campaign, channel, items):
        for label, value in (("客户", client), ("活动", campaign), ("渠道", channel)):
            if not isinstance(value, str) or not value.strip() or len(value) > 100:
                raise ValueError(f"{label}名称需为1～100字。")
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH_ITEMS:
            raise ValueError(f"每批需包含1～{MAX_BATCH_ITEMS}条素材。")
        prepared = []
        seen = set()
        for index, item in enumerate(items, 1):
            if not isinstance(item, dict):
                raise ValueError("批次素材格式无效。")
            external_id = str(item.get("external_id", index)).strip()
            text = item.get("text", "")
            evidence = item.get("evidence", "")
            images = item.get("images", [])
            if not external_id or len(external_id) > 100 or external_id in seen:
                raise ValueError("素材编号不能为空、重复或超过100字。")
            if not isinstance(text, str) or len(text) > 20000 or not isinstance(evidence, str) or len(evidence) > 20000:
                raise ValueError("文案或补充资料超过20000字。")
            if not isinstance(images, list) or len(images) > 4 or not text.strip() and not images:
                raise ValueError("每条素材需有文案或1～4张图片。")
            if item.get("mode", "ai") not in ("ai", "local") or type(item.get("complete", False)) is not bool:
                raise ValueError("审核方式或材料完整性格式无效。")
            assets = []
            total_bytes = 0
            for image in images:
                if not isinstance(image, dict) or not isinstance(image.get('filename'), str) or not isinstance(image.get('image'), str):
                    raise ValueError("图片材料格式无效。")
                try:
                    header, encoded = image['image'].split(',', 1)
                    mime = header.removeprefix('data:').removesuffix(';base64')
                    if header != f'data:{mime};base64' or mime not in ('image/png', 'image/jpeg', 'image/webp', 'application/pdf'):
                        raise ValueError()
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error):
                    raise ValueError("图片材料编码无效。")
                total_bytes += len(raw)
                if not raw or total_bytes > 8 * 1024 * 1024:
                    raise ValueError("单条图片材料合计不能超过8 MB。")
                assets.append((image['filename'][:200], mime, raw))
            seen.add(external_id)
            prepared.append((external_id, dict(text=text, evidence=evidence, images=images,
                                               complete=item.get("complete", False), mode=item.get("mode", "ai")), assets))
        batch_id = secrets.token_urlsafe(16)
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            pending = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            today = datetime.now(timezone.utc).date().isoformat()
            submitted = db.execute("SELECT count(*) FROM jobs JOIN batches ON jobs.batch_id=batches.id WHERE batches.created_at >= ?", (today,)).fetchone()[0]
            if pending + len(items) > MAX_PENDING_ITEMS:
                raise ValueError(f"演示站最多同时排队或处理{MAX_PENDING_ITEMS}条素材，请等待当前批次完成。")
            if submitted + len(items) > MAX_DAILY_ITEMS:
                raise ValueError(f"演示站每天最多提交{MAX_DAILY_ITEMS}条素材，请明天再试。")
            db.execute("INSERT INTO batches VALUES (?,?,?,?,?)", (batch_id, client.strip(), campaign.strip(), channel.strip(), now))
            for pos, (ext, payload, assets) in enumerate(prepared, 1):
                job_id = secrets.token_urlsafe(16)
                db.execute("INSERT INTO jobs (id,batch_id,external_id,position,payload,status,updated_at) VALUES (?,?,?,?,?,'queued',?)",
                           (job_id, batch_id, ext, pos, json.dumps(payload, ensure_ascii=False), now))
                db.executemany("INSERT INTO source_assets VALUES (?,?,?,?,?)",
                               [(job_id, source_pos, filename, mime, raw)
                                for source_pos, (filename, mime, raw) in enumerate(assets, 1)])
        return batch_id

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id,batch_id,payload FROM jobs WHERE status='queued' ORDER BY updated_at,position LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',attempts=attempts+1,updated_at=? WHERE id=?", (utc_now(), row["id"]))
            return dict(row)

    def finish(self, job_id, report=None, error=None):
        status = "succeeded" if report is not None else "failed"
        with self.connect() as db:
            if report is not None:
                # Drop embedded source images after OCR; the report retains recognized text.
                raw = json.loads(db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()["payload"])
                raw.pop("images", None)
                db.execute("UPDATE jobs SET payload=?,status=?,report=?,error=NULL,updated_at=? WHERE id=?",
                           (json.dumps(raw, ensure_ascii=False), status, json.dumps(report, ensure_ascii=False), utc_now(), job_id))
            else:
                db.execute("UPDATE jobs SET status=?,error=?,updated_at=? WHERE id=?",
                           (status, (error or "处理失败，请重试。")[:300], utc_now(), job_id))

    def retry(self, batch_id, job_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute("SELECT status FROM jobs WHERE id=? AND batch_id=?", (job_id, batch_id)).fetchone()
            if state is None or state['status'] != 'failed':
                return False
            pending = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if pending >= MAX_PENDING_ITEMS:
                raise ValueError("当前待处理素材已达演示站上限，请等待后重试。")
            changed = db.execute("UPDATE jobs SET status='queued',error=NULL,updated_at=? WHERE id=? AND batch_id=? AND status='failed'",
                                 (utc_now(), job_id, batch_id)).rowcount
        return bool(changed)

    def summary(self, batch_id):
        with self.connect() as db:
            batch = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
            if batch is None:
                return None
            jobs = db.execute("SELECT jobs.id,external_id,position,status,error,attempts,updated_at,COALESCE(review_decisions.decision,'pending') AS decision FROM jobs LEFT JOIN review_decisions ON review_decisions.job_id=jobs.id WHERE batch_id=? ORDER BY position", (batch_id,)).fetchall()
        rows = [dict(row) for row in jobs]
        counts = {state: sum(row["status"] == state for row in rows) for state in ("queued", "running", "succeeded", "failed")}
        return dict(batch=dict(batch), counts=counts, total=len(rows), items=rows)

    def list_recent(self, limit=20):
        self.purge_expired()
        with self.connect() as db:
            ids = [row["id"] for row in db.execute("SELECT id FROM batches ORDER BY created_at DESC LIMIT ?", (limit,))]
        return [self.summary(batch_id) for batch_id in ids]

    def detail(self, batch_id, job_id):
        with self.connect() as db:
            row = db.execute("SELECT id,external_id,position,status,payload,report,error,attempts,updated_at FROM jobs WHERE id=? AND batch_id=?",
                             (job_id, batch_id)).fetchone()
            assets = db.execute("SELECT position,filename,mime FROM source_assets WHERE job_id=? ORDER BY position", (job_id,)).fetchall()
            decision = db.execute("SELECT decision,reviewer,reason,decided_at FROM review_decisions WHERE job_id=?", (job_id,)).fetchone()
            history = db.execute("SELECT decision,reviewer,reason,decided_at FROM review_events WHERE job_id=? ORDER BY id DESC LIMIT 20", (job_id,)).fetchall()
        if row is None:
            return None
        item = dict(row)
        item["report"] = json.loads(item["report"]) if item["report"] else None
        payload = json.loads(item.pop('payload'))
        item['source'] = dict(text=payload.get('text', ''), evidence=payload.get('evidence', ''),
                              complete=payload.get('complete', False), assets=[dict(asset) for asset in assets])
        item['decision'] = dict(decision) if decision else None
        item['review_history'] = [dict(event) for event in history]
        return item

    def asset(self, batch_id, job_id, position):
        with self.connect() as db:
            row = db.execute("SELECT source_assets.filename,source_assets.mime,source_assets.content FROM source_assets JOIN jobs ON jobs.id=source_assets.job_id WHERE jobs.batch_id=? AND jobs.id=? AND source_assets.position=?",
                             (batch_id, job_id, position)).fetchone()
        return dict(row) if row else None

    def list_queue(self, search='', machine='all', decision='all', risk='all', limit=100):
        if len(search) > 100 or machine not in ('all','queued','running','succeeded','failed') or decision not in ('all','pending','approved','changes_requested','evidence_requested') or risk not in ('all','high','medium','low','none','unavailable'):
            raise ValueError('队列筛选条件无效。')
        self.purge_expired()
        with self.connect() as db:
            rows = db.execute("SELECT jobs.id AS job_id,jobs.batch_id,jobs.external_id,jobs.status,jobs.error,jobs.updated_at,json_extract(jobs.payload,'$.text') AS source_text,json_extract(jobs.payload,'$.mode') AS mode,jobs.report,batches.client,batches.campaign,batches.channel,batches.created_at,COALESCE(review_decisions.decision,'pending') AS decision FROM jobs JOIN batches ON batches.id=jobs.batch_id LEFT JOIN review_decisions ON review_decisions.job_id=jobs.id ORDER BY batches.created_at DESC,jobs.position DESC LIMIT 500").fetchall()
        output = []
        needle = search.strip().casefold()
        for row in rows:
            item = dict(row)
            if machine != 'all' and item['status'] != machine or decision != 'all' and item['decision'] != decision:
                continue
            if needle and not any(needle in str(item[key]).casefold() for key in ('client','campaign','channel','external_id')):
                continue
            report = json.loads(item.pop('report')) if item['report'] else None
            item['text_preview'] = (item.pop('source_text') or '')[:100]
            item['mode'] = item['mode'] or 'ai'
            item['conclusion'] = report.get('conclusion','') if report else ''
            item['risk'] = report_risk(report)
            if risk != 'all' and item['risk'] != risk:
                continue
            output.append(item)
        priority = {'high':0,'medium':1,'low':2,'none':3,'unavailable':4}
        output.sort(key=lambda item: (item['decision'] != 'pending', priority[item['risk']], item['created_at']), reverse=False)
        return output[:limit]

    def record_decision(self, batch_id, job_id, decision, reviewer, reason):
        if decision not in ('approved','changes_requested','evidence_requested'):
            raise ValueError('人工结论无效。')
        if not isinstance(reviewer,str) or not 2 <= len(reviewer.strip()) <= 60:
            raise ValueError('请填写2～60字的审核人名称。')
        if not isinstance(reason,str) or not 3 <= len(reason.strip()) <= 1000:
            raise ValueError('请填写3～1000字的处理依据。')
        now = utc_now()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM jobs WHERE id=? AND batch_id=?', (job_id,batch_id)).fetchone()
            if row is None:
                return False
            if row['status'] != 'succeeded':
                raise ValueError('仅已完成机器初筛的素材可以记录人工结论。')
            db.execute('INSERT INTO review_decisions(job_id,decision,reviewer,reason,decided_at) VALUES (?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET decision=excluded.decision,reviewer=excluded.reviewer,reason=excluded.reason,decided_at=excluded.decided_at',
                       (job_id,decision,reviewer.strip(),reason.strip(),now))
            db.execute('INSERT INTO review_events(job_id,decision,reviewer,reason,decided_at) VALUES (?,?,?,?,?)',
                       (job_id,decision,reviewer.strip(),reason.strip(),now))
        return True


def report_risk(report):
    if report is None:
        return 'unavailable'
    findings = report.get('findings', [])
    if report.get('blocked') or any(item.get('level') == '高' for item in findings):
        return 'high'
    if any(item.get('status') in ('疑似风险','证据不足') for item in findings):
        return 'medium'
    if findings:
        return 'low'
    return 'none'


def process_item(payload, configuration, store=None, batch_id=None):
    data = dict(text=payload["text"], evidence=payload["evidence"], complete=payload["complete"], mode=payload["mode"])
    cfg = configuration()
    material = None
    cache_hit = False
    if payload.get("images"):
        files = []
        for image in payload["images"]:
            encoded = image["image"].split(",", 1)[1]
            files.append((image["filename"], base64.b64decode(encoded, validate=True)))
        digest = hashlib.sha256(b''.join(len(raw).to_bytes(8, 'big') + raw for _, raw in files)).hexdigest()
        if store is not None and batch_id is not None:
            material = store.cached_ocr(batch_id, digest)
            cache_hit = material is not None
        if material is None:
            material = extract_bundle(files, cfg)
            if store is not None and batch_id is not None:
                store.save_ocr(batch_id, digest, material)
        else:
            # Excel row numbers differ even when the same creative is reused.
            material['filename'] = ' / '.join(name for name, _ in files)
            for page, (name, _) in zip(material['pages'], files):
                page['source_filename'] = name
        sections, quality, visual = [], [], []
        for page in material["pages"]:
            sections.append(f"\n\n【材料序号{page['page']} · 文件 {page['source_filename']} · 第{page['source_page']}页】\n" + page["text"])
            quality.extend(f"第{page['page']}页：{issue}" for issue in page["quality_issues"])
            visual.append(dict(page=page["page"], observations=page["visual_notes"]))
        data["text"] += "".join(sections)
        cfg["_quality_issues"] = quality
        cfg["_visual_context"] = visual
    started = time.monotonic()
    result = review(data, cfg)
    result["review_seconds"] = round(time.monotonic() - started, 1)
    if material:
        for finding in result["findings"]:
            if finding["quote"]:
                pages = [f"{p['source_filename']} · 第{p['source_page']}页" for p in material["pages"] if finding["quote"] in p["text"]]
                if pages:
                    finding["location"] = "、".join(pages) + "（识别文本匹配，需核对原图）"
        result.setdefault("notices", []).append("批量识别文字尚未人工核对，请对照原图检查。")
        result["material"] = material
        result["extraction_cache_hit"] = cache_hit
    return result


class BatchRunner:
    def __init__(self, store, configuration, slots, workers=2):
        self.store, self.configuration, self.slots = store, configuration, slots
        self.workers = workers
        self.wakeup = threading.Event()
        self.stopping = threading.Event()
        self.threads = []
        self.started = False
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.started:
                return
            self.started = True
            for index in range(self.workers):
                thread = threading.Thread(target=self._loop, daemon=True, name=f"batch-review-{index+1}")
                self.threads.append(thread)
                thread.start()

    def stop(self):
        self.stopping.set()
        self.wakeup.set()
        for thread in self.threads:
            thread.join(timeout=2)

    def notify(self):
        self.start()
        self.wakeup.set()

    def _loop(self):
        while not self.stopping.is_set():
            try:
                job = self.store.claim()
                if job is None:
                    self.wakeup.wait(timeout=5)
                    self.wakeup.clear()
                    continue
                try:
                    with self.slots:
                        result = process_item(json.loads(job["payload"]), self.configuration, self.store, job["batch_id"])
                    self.store.finish(job["id"], report=result)
                except Exception as exc:
                    message = str(exc) if isinstance(exc, ValueError) else "模型或服务处理失败，请检查配置后重试。"
                    self.store.finish(job["id"], error=message)
            except Exception:
                # Keep the worker alive if SQLite or another infrastructure call fails.
                time.sleep(2)
