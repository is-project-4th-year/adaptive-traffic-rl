import traci, sys

SUMO = ["sumo", "-n", "cross.net.xml", "-r", "typical.rou.xml",
        "--start", "true", "--quit-on-end", "true", "--step-length", "1"]
traci.start(SUMO)

tls = traci.trafficlight.getIDList()
if not tls: sys.exit("No TLS found.")
tid = tls[0]

# controlledLinks: list of cols; each col is a list of (inLane, outLane, via) tuples for that signal index
cl = traci.trafficlight.getControlledLinks(tid)
n = len(cl)

def in_edge(lane_id):
    # lane_id looks like "N_in_0"; edge is everything before the last '_' index
    return lane_id.rsplit('_', 1)[0]

# Build phase masks by which inbound edge they come from
ns_green = ['r'] * n
ew_green = ['r'] * n

for i, col in enumerate(cl):
    # pick the first (inLane, outLane, via) tuple for the controlled link
    if not col: continue
    inLane, outLane, via = col[0]
    e = in_edge(inLane)
    if e in ("N_in", "S_in"):
        ns_green[i] = 'G'
    if e in ("E_in", "W_in"):
        ew_green[i] = 'G'

ns_yellow = ['y' if c=='G' else 'r' for c in ns_green]
ew_yellow = ['y' if c=='G' else 'r' for c in ew_green]

def s(lst): return "".join(lst)

xml = f"""<additional>
  <tlLogic id="{tid}" type="static" programID="baseline" offset="0">
    <phase duration="30" state="{s(ns_green)}"/>
    <phase duration="4"  state="{s(ns_yellow)}"/>
    <phase duration="30" state="{s(ew_green)}"/>
    <phase duration="4"  state="{s(ew_yellow)}"/>
  </tlLogic>
</additional>
"""
open("cross.tll.xml","w").write(xml)
print("Wrote cross.tll.xml with", n, "signals for", tid)
traci.close()
