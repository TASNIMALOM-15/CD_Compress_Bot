"""Telegram video compressor bot (FFmpeg + Pyrogram/pyrofork).

Send a video -> choose resolution / FPS / quality -> Start.
One video is compressed at a time (low RAM); the rest wait in a queue.
Jobs are saved to disk, so after a crash the bot resumes them automatically.
"""
import asyncio
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass

import aiohttp
from aiohttp import web
from pyrogram import Client, filters, idle
from pyrogram.errors import FloodWait, MessageNotModified
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# ----------------------------------------------------------------- settings
API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
BOT_TOKEN = os.environ["BOT_TOKEN"]
ALLOWED_USERS = {int(x) for x in os.environ.get("ALLOWED_USERS", "").replace(" ", "").split(",") if x}
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "20"))
FFMPEG_THREADS = os.environ.get("FFMPEG_THREADS", "2")
PRESET = os.environ.get("FFMPEG_PRESET", "veryfast")
PORT = int(os.environ.get("PORT", "10000"))
SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", "")
WORK_DIR = os.environ.get("WORK_DIR", "/tmp/vcbot")
JOBS_FILE = os.path.join(WORK_DIR, "jobs.json")
MAX_UPLOAD = 2000 * 1024 * 1024  # Telegram bot limit (~2 GB)

RESOLUTIONS = [360, 480, 720, 1080]
FPS_OPTIONS = [0, 24, 30, 60]  # 0 = keep original
QUALITY = {"high": ("High quality", 23), "mid": ("Balanced", 27), "small": ("Small size", 31)}

app = Client("vcbot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN, in_memory=True)


@dataclass
class Job:
    id: str
    chat_id: int
    user_id: int
    msg_id: int      # the user's original video message
    status_id: int   # the bot's status message (edited with progress)
    height: int
    fps: int
    crf: int
    attempts: int = 0


queue: "asyncio.Queue[Job]" = None  # created in main()
pending = {}      # options-message id -> selection (before Start is pressed)
active = {}       # job id -> Job (queued or running)
cancelled = set()  # job ids cancelled by the user
current = {"job": None, "proc": None}
tasks = []


# ----------------------------------------------------------------- helpers
def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def clock(sec):
    sec = int(max(sec, 0))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def bar(frac, width=12):
    frac = min(max(frac, 0.0), 1.0)
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


class Throttle:
    def __init__(self, interval=5.0):
        self.interval, self.last = interval, 0.0

    def ready(self):
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.last = now
            return True
        return False


async def edit(chat_id, mid, text, kb=None):
    try:
        await app.edit_message_text(chat_id, mid, text, reply_markup=kb)
    except MessageNotModified:
        pass
    except FloodWait as e:
        await asyncio.sleep(min(getattr(e, "value", 5), 10))
    except Exception as e:  # never let a cosmetic edit kill a job
        print("edit failed:", e)


def cancel_kb(job):
    return InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data=f"cx:{job.id}")]])


def save_jobs():
    try:
        os.makedirs(WORK_DIR, exist_ok=True)
        tmp = JOBS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump([asdict(j) for j in active.values()], f)
        os.replace(tmp, JOBS_FILE)
    except Exception as e:
        print("save_jobs failed:", e)


def options_kb(sel):
    def mark(ok, text):
        return ("✅ " if ok else "") + text

    return InlineKeyboardMarkup([
        [InlineKeyboardButton(mark(sel["h"] == h, f"{h}p"), callback_data=f"h:{h}") for h in RESOLUTIONS],
        [InlineKeyboardButton(mark(sel["f"] == f, "Original FPS" if f == 0 else f"{f} FPS"),
                              callback_data=f"f:{f}") for f in FPS_OPTIONS],
        [InlineKeyboardButton(mark(sel["q"] == k, v[0]), callback_data=f"q:{k}") for k, v in QUALITY.items()],
        [InlineKeyboardButton("🚀 Start", callback_data="go")],
    ])


def options_text(sel):
    fps = "Original" if sel["f"] == 0 else sel["f"]
    return (f"🎞 Choose compression settings\n\nResolution: {sel['h']}p\nFPS: {fps}\n"
            f"Quality: {QUALITY[sel['q']][0]}\n\nThen press 🚀 Start.")


# ----------------------------------------------------------------- ffmpeg
def parse_rate(text):
    try:
        a, b = text.split("/")
        return float(a) / float(b) if float(b) else 0.0
    except Exception:
        return 0.0


async def probe(path):
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    data = json.loads(out or b"{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("No video stream found in this file")
    w, h = int(video["width"]), int(video["height"])
    rot = 0
    try:
        rot = int(float(video.get("tags", {}).get("rotate", 0)))
    except Exception:
        pass
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = parse_rate(video.get("avg_frame_rate", "")) or parse_rate(video.get("r_frame_rate", ""))
    dur = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    return {"w": w, "h": h, "fps": fps, "dur": dur}


async def run_ffmpeg(job, src, dst, info, workdir):
    target = job.height
    w, h = info["w"], info["h"]
    if min(w, h) > target:  # never upscale
        if w >= h:
            vf = f"scale=-2:{target}"
            ow, oh = round(w * target / h / 2) * 2, target
        else:
            vf = f"scale={target}:-2"
            ow, oh = target, round(h * target / w / 2) * 2
    else:
        vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        ow, oh = w // 2 * 2, h // 2 * 2
    cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", src,
           "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn", "-vf", vf]
    out_fps = info["fps"]
    if job.fps and info["fps"] > job.fps + 0.5:  # never raise FPS
        cmd += ["-r", str(job.fps)]
        out_fps = job.fps
    cmd += ["-c:v", "libx264", "-preset", PRESET, "-crf", str(job.crf), "-pix_fmt", "yuv420p",
            "-threads", FFMPEG_THREADS, "-c:a", "aac", "-b:a", "96k", "-ac", "2",
            "-movflags", "+faststart", "-max_muxing_queue_size", "1024",
            "-progress", "pipe:1", "-nostats", dst]

    log = open(os.path.join(workdir, "ffmpeg.log"), "wb")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=log)
    current["proc"] = proc
    started, thr = time.monotonic(), Throttle(5.0)
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="ignore").strip()
            if text.startswith(("out_time_us=", "out_time_ms=")) and info["dur"] > 0:
                try:
                    done = int(text.split("=")[1]) / 1_000_000
                except ValueError:
                    continue
                frac = done / info["dur"]
                if thr.ready() and not job.id in cancelled:
                    el = time.monotonic() - started
                    eta = el / frac * (1 - frac) if frac > 0.01 else 0
                    await edit(job.chat_id, job.status_id,
                               f"🗜 Compressing...\n{bar(frac)} {min(int(frac * 100), 99)}%\n"
                               f"⏱ {clock(el)} elapsed • ETA {clock(eta)}", cancel_kb(job))
        await proc.wait()
    finally:
        current["proc"] = None
        log.close()
    if job.id in cancelled:
        return None
    if proc.returncode != 0:
        tail = open(os.path.join(workdir, "ffmpeg.log"), errors="ignore").read()[-400:]
        raise RuntimeError(f"FFmpeg failed:\n{tail}")
    return {"w": ow, "h": oh, "fps": out_fps}


# ----------------------------------------------------------------- job pipeline
async def process(job):
    workdir = os.path.join(WORK_DIR, job.id)
    os.makedirs(workdir, exist_ok=True)
    started = time.monotonic()
    try:
        msg = await app.get_messages(job.chat_id, job.msg_id)
        media = msg.video or msg.document or msg.animation if msg else None
        if not media:
            raise RuntimeError("Original video message not found. Please send the video again.")

        # 1) download
        thr = Throttle(5.0)

        async def dl_cb(cur, total):
            if job.id in cancelled:
                app.stop_transmission()
            if thr.ready() and total:
                await edit(job.chat_id, job.status_id,
                           f"⬇️ Downloading...\n{bar(cur / total)} {int(cur * 100 / total)}%\n"
                           f"{human(cur)} / {human(total)}", cancel_kb(job))

        src = await app.download_media(msg, file_name=os.path.join(workdir, "input.bin"), progress=dl_cb)
        if job.id in cancelled:
            return
        if not src:
            raise RuntimeError("Download failed")
        orig_size = os.path.getsize(src)

        # 2) probe + compress
        info = await probe(src)
        dst = os.path.join(workdir, "output.mp4")
        res = await run_ffmpeg(job, src, dst, info, workdir)
        if res is None or job.id in cancelled:
            return
        os.remove(src)  # free disk before upload
        new_size = os.path.getsize(dst)
        if new_size > MAX_UPLOAD:
            raise RuntimeError(f"Result is {human(new_size)}, above Telegram's 2 GB bot limit. Try a lower setting.")

        # 3) thumbnail (optional)
        thumb = os.path.join(workdir, "thumb.jpg")
        t = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-hide_banner", "-nostdin", "-ss", "1" if info["dur"] > 2 else "0", "-i", dst,
            "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "5", thumb,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await t.wait()
        if not os.path.exists(thumb):
            thumb = None

        # 4) upload
        thr2 = Throttle(5.0)

        async def up_cb(cur, total):
            if job.id in cancelled:
                app.stop_transmission()
            if thr2.ready() and total:
                await edit(job.chat_id, job.status_id,
                           f"⬆️ Uploading...\n{bar(cur / total)} {int(cur * 100 / total)}%\n"
                           f"{human(cur)} / {human(total)}", cancel_kb(job))

        saved = (1 - new_size / orig_size) * 100 if orig_size else 0
        note = "" if new_size < orig_size else "\n⚠️ Result is not smaller. Try a lower resolution or 'Small size'."
        caption = (f"🎬 {res['h'] if res['w'] >= res['h'] else res['w']}p • {res['fps']:.0f} FPS\n"
                   f"📦 {human(orig_size)} → {human(new_size)} ({saved:+.0f}%){note}")
        base = os.path.splitext(getattr(media, "file_name", None) or "video")[0]
        await app.send_video(job.chat_id, dst, caption=caption, duration=int(info["dur"]),
                             width=res["w"], height=res["h"], thumb=thumb, supports_streaming=True,
                             file_name=f"{base}_compressed.mp4", progress=up_cb)
        if job.id in cancelled:
            return
        await edit(job.chat_id, job.status_id,
                   f"✅ Done in {clock(time.monotonic() - started)}\n"
                   f"{human(orig_size)} → {human(new_size)} ({saved:+.0f}%)")
    except Exception as e:
        if job.id not in cancelled:
            print("job failed:", e)
            await edit(job.chat_id, job.status_id, f"❌ Failed\n{str(e)[:700]}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        active.pop(job.id, None)
        cancelled.discard(job.id)
        save_jobs()


async def worker():
    while True:
        job = await queue.get()
        try:
            if job.id in cancelled or job.id not in active:
                cancelled.discard(job.id)
                continue
            current["job"] = job
            await process(job)
        except Exception as e:
            print("worker error:", e)
        finally:
            current["job"] = None
            queue.task_done()


async def recover_jobs():
    """After a crash/restart: wipe leftovers, then re-queue the saved jobs."""
    saved = []
    try:
        with open(JOBS_FILE) as f:
            saved = json.load(f)
    except Exception:
        pass
    for name in os.listdir(WORK_DIR):
        if name != "jobs.json":
            p = os.path.join(WORK_DIR, name)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    for d in saved:
        try:
            job = Job(**d)
        except Exception:
            continue
        job.attempts += 1
        if job.attempts > 2:
            await edit(job.chat_id, job.status_id, "❌ Failed after restart. Please send the video again.")
            continue
        active[job.id] = job
        await edit(job.chat_id, job.status_id, "♻️ Bot restarted. Resuming your video...", cancel_kb(job))
        await queue.put(job)
    save_jobs()


# ----------------------------------------------------------------- telegram handlers
def allowed(user_id):
    return not ALLOWED_USERS or user_id in ALLOWED_USERS


@app.on_message(filters.private & filters.command(["start", "help"]))
async def start_cmd(_, m):
    if not allowed(m.from_user.id):
        return
    await m.reply_text("👋 Send me a video (or a video file). I will ask for resolution, FPS and quality, "
                       "then compress it.\n\n/queue - show waiting videos")


@app.on_message(filters.private & filters.command("queue"))
async def queue_cmd(_, m):
    if not allowed(m.from_user.id):
        return
    mine = sum(1 for j in active.values() if j.user_id == m.from_user.id)
    await m.reply_text(f"📋 Waiting: {queue.qsize()}\n▶️ Running: {1 if current['job'] else 0}\n👤 Yours: {mine}")


@app.on_message(filters.private & (filters.video | filters.document | filters.animation))
async def on_video(_, m):
    if not allowed(m.from_user.id):
        return
    if m.document and not (m.document.mime_type or "").startswith("video/"):
        return
    if len(active) >= MAX_QUEUE:
        await m.reply_text("⚠️ The queue is full. Please try again later.")
        return
    sel = {"h": 480, "f": 0, "q": "mid", "msg": m.id, "user": m.from_user.id}
    sent = await app.send_message(m.chat.id, options_text(sel), reply_markup=options_kb(sel))
    pending[sent.id] = sel


@app.on_callback_query()
async def on_button(_, cq):
    data = cq.data or ""
    chat_id, mid = cq.message.chat.id, cq.message.id

    if data.startswith("cx:"):
        job = active.get(data[3:])
        if not job or job.user_id != cq.from_user.id:
            await cq.answer("Not found", show_alert=True)
            return
        cancelled.add(job.id)
        active.pop(job.id, None)
        save_jobs()
        proc = current["proc"]
        if current["job"] and current["job"].id == job.id and proc:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await edit(chat_id, mid, "❌ Cancelled")
        await cq.answer("Cancelled")
        return

    sel = pending.get(mid)
    if not sel:
        await cq.answer("This menu expired. Please send the video again.", show_alert=True)
        return
    if sel["user"] != cq.from_user.id:
        await cq.answer("Not your video", show_alert=True)
        return

    if data.startswith("h:"):
        sel["h"] = int(data[2:])
    elif data.startswith("f:"):
        sel["f"] = int(data[2:])
    elif data.startswith("q:") and data[2:] in QUALITY:
        sel["q"] = data[2:]
    elif data == "go":
        if len(active) >= MAX_QUEUE:
            await cq.answer("Queue is full", show_alert=True)
            return
        pending.pop(mid, None)
        job = Job(id=uuid.uuid4().hex[:10], chat_id=chat_id, user_id=sel["user"], msg_id=sel["msg"],
                  status_id=mid, height=sel["h"], fps=sel["f"], crf=QUALITY[sel["q"]][1])
        active[job.id] = job
        save_jobs()
        pos = queue.qsize() + (1 if current["job"] else 0)
        await queue.put(job)
        await edit(chat_id, mid, f"⏳ Queued (position {pos + 1})", cancel_kb(job))
        await cq.answer("Added to queue")
        return
    await edit(chat_id, mid, options_text(sel), options_kb(sel))
    await cq.answer()


# ----------------------------------------------------------------- web server (Render needs an open port)
async def health(_):
    return web.Response(text="OK")


async def self_ping():
    """Render Free sleeps after 15 min without web traffic: ping ourselves."""
    if not SELF_URL:
        return
    while True:
        await asyncio.sleep(240)
        try:
            async with aiohttp.ClientSession() as s:
                await s.get(SELF_URL, timeout=aiohttp.ClientTimeout(total=20))
        except Exception as e:
            print("self ping failed:", e)


async def main():
    global queue
    queue = asyncio.Queue()
    os.makedirs(WORK_DIR, exist_ok=True)
    web_app = web.Application()
    web_app.router.add_get("/", health)
    runner = web.AppRunner(web_app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()

    await app.start()
    me = await app.get_me()
    print(f"Bot started: @{me.username}")
    await recover_jobs()
    tasks.append(asyncio.create_task(worker()))
    tasks.append(asyncio.create_task(self_ping()))
    await idle()
    await app.stop()


if __name__ == "__main__":
    app.run(main())
