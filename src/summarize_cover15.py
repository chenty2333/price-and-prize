"""Progress / final verdict of the independent fifteen-good run (src/cover15_ours.cpp in orbit mode).

Reads results/cover15_ours/worker_*.log (one line per checked orbit representative, written to stderr by the program)
and worker_*.json (one JSON line printed at the end of a worker's run).  The run is complete and successful iff
  * every worker has printed its JSON line, and
  * the sum of "uncovered" is 0, and
  * the orbit weights of all representatives sum to 1,401,400 (= the number of partitions of 15 goods into 5 triples).
Usage: python3 src/summarize_cover15.py [results/cover15_ours]
"""
import glob
import json
import re
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results/cover15_ours"
reps = weight = bad = 0
finished, total_workers = 0, 0
uncovered_json = 0
for f in sorted(glob.glob(f"{d}/worker_*.log")):
    total_workers += 1
    for line in open(f):
        mo = re.match(r"P (\d+) w=(\d+) n=\d+ calls=\d+ states=\d+ flushes=\d+ sec=[\d.]+ ok=(\d)", line)
        if mo:
            reps += 1
            weight += int(mo.group(2))
            bad += (mo.group(3) == "0")
    try:
        last = open(f.replace(".log", ".json")).read().strip().splitlines()[-1]
        j = json.loads(last)
        finished += 1
        uncovered_json += j["uncovered"]
    except Exception:
        pass
print(f"workers finished: {finished}/{total_workers}; representatives checked: {reps}; total orbit weight: {weight} / 1401400; "
      f"uncovered (log lines): {bad}; uncovered (worker summaries): {uncovered_json}")
if finished == total_workers and total_workers > 0:
    print("COMPLETE and", "SUCCESSFUL" if (bad == 0 and uncovered_json == 0 and weight == 1401400) else "NOT SUCCESSFUL: inspect the logs")
else:
    print("still running")
