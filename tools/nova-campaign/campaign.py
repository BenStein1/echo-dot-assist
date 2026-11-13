#!/usr/bin/env python3
"""Hands-free wake-word recording campaign for the Echo Dot ("Nova").

  campaign.py plan              write build/nova-campaign/plan.json (deterministic; blocks A1.. train/test, F1.. final)
  campaign.py record BLOCK      run one block at the Dot: it shows what to say, you say it, no keys to press
  campaign.py process [BLOCK..] align speech, extract native features, cut windows, write train/test manifests
  campaign.py status            what is recorded / processed

Recordings are the Dot's OWN processed mic stream (SIGTTIN dump), the exact audio the live engine hears.
Keep the software Mute ON while recording so the Dot never answers over you.  Mute stops the Dot acting on the
wake word; it does not stop the stream (the script checks that the capture really has audio).
All outputs stay under build/ (ignored by git): personal voice recordings never get committed.
"""
import argparse, hashlib, json, math, os, pathlib, random, struct, subprocess, sys, time, wave, array
import concurrent.futures

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "build" / "nova-campaign"
ADB = ["adb", "-s", os.environ.get("DOT", "10.0.0.199:5555")]
REMOTE = "/data/local/hassmic/state/capture.raw"
RATE = 16000
STOCK = ROOT / "firmware/donut/rootfs/system/local/models/keyword/en-US/ALEXA/pryon.manifest"
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import phrases as P

STYLES = [
    "normal voice, arm's length", "normal voice, across the room", "quietly", "whisper",
    "loud, like calling from the next room", "fast and casual", "slow and drawn out", "tired, mumbling",
    "excited", "facing away from the Dot", "walking past the Dot", "annoyed", "as a question", "flat, monotone",
]


def shell(cmd, timeout=20):
    return subprocess.check_output(ADB + ["shell", cmd], text=True, timeout=timeout).strip()


def slot_len(text, kind):
    if kind == "quiet":
        return 6.0
    words = max(1, len(text.split()))
    return min(8.0, 2.2 + 0.45 * words)


def make_plan():
    rnd = random.Random(20261112)
    plan = {"blocks": {}}

    def prompts(pos, neg, speech, noise, quiets, split_rule, tag):
        items = []
        for t in pos: items.append(("pos", t))
        for t in neg: items.append(("neg", t))
        for t in speech:
            if "nova" not in t.lower(): items.append(("neg", t))
        for t in noise: items.append(("noise", t))
        for _ in range(quiets): items.append(("quiet", "stay silent, normal room noise"))
        rnd.shuffle(items)
        # each prompt: balanced style rotation, independent of kind
        styles = (STYLES * (len(items) // len(STYLES) + 1))[:len(items)]
        rnd.shuffle(styles)
        out = []
        for i, ((kind, text), style) in enumerate(zip(items, styles)):
            if kind in ("noise", "quiet"): style = ""
            out.append({"id": f"{tag}-{i:03d}", "kind": kind, "text": text, "style": style,
                        "split": split_rule(kind, text, i, tag), "dur": slot_len(text, kind)})
        return out

    # TRAIN/TEST pool: each positive and negative phrase is said in several blocks with different styles; the TEST
    # side holds out whole phrases (by hash) so the same text is never in both splits.
    def tt(kind, text, i, tag):
        if kind == "pos": return "test" if tag in ("A4", "A8") else "train"     # positives held out by whole block
        if kind in ("noise", "quiet"): return "test" if i % 5 == 0 else "train"
        return "test" if int(hashlib.sha256(text.casefold().encode()).hexdigest(), 16) % 5 == 0 else "train"

    pos_rep = P.POS_TRAIN
    for n in range(1, 5):                     # A1..A4: the whole positive pool, a quarter of the confusables each
        neg = P.NEG_CONFUSABLE_TRAIN[(n - 1)::4]
        speech = P.NEG_SPEECH_TRAIN[(n - 1)::4]
        noise = P.NOISE[(n - 1)::4]
        pos = pos_rep
        plan["blocks"][f"A{n}"] = {"prompts": prompts(pos, neg, speech, noise, 3, tt, f"A{n}")}
    for n in range(5, 9):                     # A5..A8: second pass, other styles; confusables again (dot heavy)
        dots = [t for t in P.NEG_CONFUSABLE_TRAIN if "dot" in t.lower()]
        neg = P.NEG_CONFUSABLE_TRAIN[(n - 5)::4] + dots
        speech = P.NEG_SPEECH_TRAIN[(n - 5)::4]
        noise = P.NOISE[(n - 5)::4]
        pos = pos_rep[(n - 5)::2] + ["Nova", "Nova?", "Nova, what time is it"]
        plan["blocks"][f"A{n}"] = {"prompts": prompts(pos, neg, speech, noise, 3, tt, f"A{n}")}
    for n in (1, 2):                          # F1, F2: final held-out gate (disjoint pools, record on another day)
        half = lambda L: L[(n - 1)::2]
        plan["blocks"][f"F{n}"] = {"prompts": prompts(half(P.POS_FINAL), half(P.NEG_CONFUSABLE_FINAL),
                                                      half(P.NEG_SPEECH_FINAL), half(P.NOISE_FINAL), 4,
                                                      lambda k, t, i, tag: "final", f"F{n}")}
    plan["blocks"]["R1"] = {"stream": {"text": " ".join(P.READ_PARAGRAPHS), "split": "train",
                                       "note": "Read these three paragraphs aloud at normal pace, with the Dot beside you.",
                                       "paragraphs": P.READ_PARAGRAPHS}}
    plan["blocks"]["M1"] = {"stream": {"text": "(ambient media)", "split": "test", "minutes": 12,
                                       "note": "Play TV / a podcast / music with talking at normal volume near the Dot. Stay quiet."}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    for k, b in plan["blocks"].items():
        if "prompts" in b:
            ps = b["prompts"]; secs = sum(p["dur"] + 2.2 for p in ps)
            print(f"{k}: {len(ps)} prompts, {sum(p['kind']=='pos' for p in ps)} Nova, ~{secs/60:.1f} min")
        else:
            print(f"{k}: stream")


def pids():
    out = []
    for p in shell("pidof hassmic").split():
        if shell(f"cat /proc/{p}/comm") == "hassmic": out.append(p)
    if len(out) != 1: raise SystemExit(f"Expected one hassmic daemon, found {out}")
    return out[0]


def record(block):
    plan = json.loads((OUT / "plan.json").read_text())
    b = plan["blocks"][block]
    d = OUT / "raw" / block
    if d.exists(): raise SystemExit(f"{d} already exists; move it aside to re-record this block")
    pid = pids()
    rec = lambda: REMOTE in shell(f"ls -l /proc/{pid}/fd")
    if rec(): raise SystemExit("A capture is already running on the Dot; left untouched.")
    print("\nMake sure software Mute is ON (Home Assistant switch) so the Dot does not answer.")
    input("Stand where you would normally talk to it, then press Enter. ")
    d.mkdir(parents=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{REMOTE}.before-campaign-{stamp}"
    shell(f"if [ -e {REMOTE} ]; then mv {REMOTE} {backup}; fi")
    events = []; started = False; pulled = False; t_stop = None
    try:
        shell(f"kill -TTIN {pid}"); started = True; t0 = time.monotonic(); time.sleep(1)
        if not rec(): raise RuntimeError("capture did not open")
        print("\nStay quiet for 4 seconds (room tone)."); time.sleep(4)
        if "prompts" in b:
            ps = b["prompts"]
            for i, p in enumerate(ps):
                os.system("clear")
                print(f"  [{block}] {i+1}/{len(ps)}   ({int(sum(x['dur']+2.2 for x in ps[i:])/60+.5)} min left)\n")
                if p["kind"] == "pos": print("  SAY:   " + p["text"] + "\n")
                elif p["kind"] == "neg": print("  SAY (not a wake word):   " + p["text"] + "\n")
                elif p["kind"] == "noise": print("  MAKE THIS SOUND:   " + p["text"] + "\n")
                else: print("  STAY QUIET\n")
                if p["style"]: print("  HOW:   " + p["style"])
                sys.stdout.write("\a"); sys.stdout.flush()
                events.append(dict(p, t=time.monotonic() - t0))
                time.sleep(p["dur"] + 0.8)
                os.system("clear"); print("  ...")
                time.sleep(1.4)          # silence between prompts: keeps utterances separable
        else:
            s = b["stream"]; os.system("clear")
            print("  " + s["note"] + "\n")
            if "paragraphs" in s:
                for k, para in enumerate(s["paragraphs"]):
                    print(f"  --- paragraph {k+1} ---\n"); print("  " + para + "\n")
                    events.append({"id": f"{block}-p{k}", "kind": "stream", "text": para, "t": time.monotonic() - t0})
                    input("  (press Enter when you finish this paragraph) ")
                    events[-1]["t_end"] = time.monotonic() - t0
            else:
                secs = int(s["minutes"] * 60); events.append({"id": f"{block}-m", "kind": "stream", "text": s["text"], "t": time.monotonic() - t0})
                for left in range(secs, 0, -1):
                    sys.stdout.write(f"\r  recording... {left//60}:{left%60:02d} left "); sys.stdout.flush(); time.sleep(1)
                events[-1]["t_end"] = time.monotonic() - t0
        time.sleep(2)
    except KeyboardInterrupt:
        print("\nInterrupted; saving what was captured.")
    finally:
        (d / "events.json").write_text(json.dumps(events, indent=1))
        if started:
            if not rec(): raise RuntimeError("capture stopped on its own; backup left at " + backup)
            t_stop = time.monotonic() - t0
            shell(f"kill -TTIN {pid}"); time.sleep(1)
            if rec(): raise RuntimeError("capture still running; backup left at " + backup)
            subprocess.run(ADB + ["pull", REMOTE, str(d / "capture.raw")], check=True, timeout=120); pulled = True
            (d / "meta.json").write_text(json.dumps({"t_stop": t_stop, "block": block}))
        if pulled or not started:
            shell(f"if [ -e {backup} ]; then mv {backup} {REMOTE}; fi")
    if pulled:
        raw = (d / "capture.raw").read_bytes()
        a = array.array("h"); a.frombytes(raw[: len(raw) // 2 * 2])
        rms = math.sqrt(sum(x * x for x in a[::40]) / max(1, len(a[::40])))
        with wave.open(str(d / "capture.wav"), "wb") as w:
            w.setparams((1, 2, RATE, 0, "NONE", "not compressed")); w.writeframes(raw)
        print(f"\nSaved {d}  ({len(raw)/32000:.0f}s, rms {rms:.0f})")
        if rms < 5: print("WARNING: the capture is silent. Hardware mute on, or the mic stream is off. Redo this block.")


def frames_rms(pcm):
    a = array.array("h"); a.frombytes(pcm[: len(pcm) // 2 * 2])
    n = RATE // 50
    return [math.sqrt(sum(x * x for x in a[i:i + n]) / n) for i in range(0, len(a) - n, n)]


def speech_span(fr, lo, hi, floor):
    thr = max(floor * 3.0, 40.0)
    idx = [i for i in range(max(0, lo), min(len(fr), hi)) if fr[i] > thr]
    if not idx: return None
    # one utterance may have pauses inside it ("Nova, ... what time is it"): take first to last frame above threshold
    s, e = idx[0], idx[-1]
    return (s / 50.0, (e + 1) / 50.0)


def build_events(block, plan_block):
    d = OUT / "raw" / block
    pcm = (d / "capture.raw").read_bytes()
    ev = json.loads((d / "events.json").read_text()); meta = json.loads((d / "meta.json").read_text())
    start_off = meta["t_stop"] - len(pcm) / 32000.0       # monotonic-since-t0 at which the file began
    fr = frames_rms(pcm); floor = sorted(fr[:200])[len(fr[:200]) // 2] if fr else 1
    out = []; skipped = []
    for e in ev:
        if e["kind"] == "stream":
            t0 = e["t"] - start_off; t1 = e["t_end"] - start_off
            if "paragraphs" in plan_block.get("stream", {}) and e is ev[0]: t0 = max(0, t0)
            pos = t0 + 2.0
            while pos + 1 < t1:
                seg_end = min(t1, pos + 8.0)
                out.append(dict(id=e["id"], label=0, split=plan_block["stream"]["split"], text=e["text"], t0=pos, t1=seg_end, speaker=f"ben:{block}"))
                pos = seg_end
            continue
        t = e["t"] - start_off
        lo, hi = int((t - 0.3) * 50), int((t + e["dur"] + 1.6) * 50)
        if e["kind"] == "quiet":
            out.append(dict(id=e["id"], label=0, split=e["split"], text="(quiet)", t0=t + 0.3, t1=min(t + e["dur"], t + 7.5), speaker=f"ben:{block}")); continue
        sp = speech_span(fr, lo, hi, floor)
        if sp is None or sp[1] - sp[0] < 0.12:
            skipped.append((e["id"], e["kind"], e["text"], "no speech/sound detected")); continue
        s, f = max(0.0, sp[0] - 0.3), sp[1] + 0.3
        if f - s > 8.5: f = s + 8.5
        label = 1 if e["kind"] == "pos" else 0
        out.append(dict(id=e["id"], label=label, split=e["split"], text=e["text"], t0=s, t1=f, speaker=f"ben:{block}", style=e.get("style", "")))
    return pcm, out, skipped


SEG = 100.0; LEAD = 9.0


def extract_one(args):
    raw, dest = args
    if dest.exists() and dest.stat().st_size > 10000: return str(dest) + " cached"
    log = dest.with_suffix(".extract.txt")
    with log.open("w") as f:
        p = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--bind", str(ROOT / "build"), str(ROOT / "build"), "--dev", "/dev",
                            "--proc", "/proc", "--unshare-pid", "--die-with-parent", "/usr/bin/qemu-arm", "-L",
                            str(ROOT / "firmware/donut/rootfs"), "-E", "LD_LIBRARY_PATH=/system/lib:/system/vendor/lib",
                            str(ROOT / "build/donut/pryon_extract"), str(STOCK), str(raw), str(dest)], stdout=f, stderr=f)
    if p.returncode: raise RuntimeError(f"{log} rc={p.returncode}")
    return str(dest) + " done"


def process(blocks):
    plan = json.loads((OUT / "plan.json").read_text())
    feat = OUT / "features"; feat.mkdir(exist_ok=True)
    jobs = []; windows = []; report = {}
    for block in blocks:
        if not (OUT / "raw" / block / "capture.raw").exists(): print(f"{block}: not recorded, skipped"); continue
        pcm, evs, skipped = build_events(block, plan["blocks"][block])
        report[block] = {"events": len(evs), "pos": sum(e["label"] for e in evs), "skipped": skipped}
        n = len(pcm) // 2; seg_n = int(SEG * RATE); lead_n = int(LEAD * RATE); s = 0; segs = []
        while True:
            a = max(0, s - lead_n); b = min(n, s + seg_n)
            name = f"{block}-seg{len(segs):02d}"; rawp = feat / (name + ".raw")
            rawp.write_bytes(b"\0" * 64000 + pcm[a * 2:b * 2] + b"\0" * 64000)   # 2 s pad each end, as prepare.py does
            segs.append((a / RATE, b / RATE, name)); jobs.append((rawp, feat / (name + ".bin")))
            if b >= n: break
            s += seg_n
        for e in evs:
            for (a, b, name) in segs:
                if e["t1"] <= b and (a == 0 or e["t0"] - a >= LEAD - 1.0):
                    windows.append(dict(source=name + ".bin", start=e["t0"] - a + 2.0, end=e["t1"] - a + 2.0, label=e["label"],
                                        split=e["split"], text=e["text"], speaker=e["speaker"], id=e["id"], block=block)); break
            else:
                report[block].setdefault("unplaced", []).append(e["id"])
    print(f"extracting {len(jobs)} segments on {min(len(jobs), 20)} workers...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        for m in ex.map(extract_one, jobs): print(m, flush=True)
    (feat / "events.json").write_text(json.dumps([{k: w[k] for k in ("source", "start", "end", "label", "split", "text", "speaker")} for w in windows], indent=1))
    (OUT / "windows-index.json").write_text(json.dumps(windows, indent=1))
    (OUT / "process-report.json").write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(f"{k}: {v['events']} events ({v['pos']} Nova), skipped {len(v['skipped'])}, unplaced {len(v.get('unplaced', []))}")
    print("events written; next: python3 tools/pryon-native/stream_windows.py build/nova-campaign/features/events.json build/nova-campaign/windows")


def status():
    plan = json.loads((OUT / "plan.json").read_text())
    for k in plan["blocks"]:
        r = (OUT / "raw" / k / "capture.raw").exists()
        print(f"{k}: {'recorded' if r else 'todo'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("blocks", nargs="*")
    a = ap.parse_args()
    if a.cmd == "plan": make_plan()
    elif a.cmd == "record": record(a.blocks[0])
    elif a.cmd == "process": process(a.blocks or [k for k in json.loads((OUT / "plan.json").read_text())["blocks"]])
    elif a.cmd == "status": status()
