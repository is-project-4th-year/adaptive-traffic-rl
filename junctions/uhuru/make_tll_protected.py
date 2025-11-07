import traci, sys

# Launch a minimal sim to query TLS structure
CMD = ["sumo", "-n", "cross.net.xml", "-r", "typical.rou.xml",
       "--start", "true", "--quit-on-end", "true", "--step-length", "1"]
traci.start(CMD)

tls = traci.trafficlight.getIDList()
if not tls:
    sys.exit("No traffic lights found.")
tid = tls[0]

# controlledLinks: list of signal indices; each has a list of tuples (inLane, outLane, via)
links = traci.trafficlight.getControlledLinks(tid)
N = len(links)

def edge_of_lane(lane_id):
    # lane ids like 'N_in_0' -> edge 'N_in'
    return lane_id.rsplit('_', 1)[0]

def mov_type(in_edge, out_edge):
    # LHT mapping at a plus intersection:
    # N_in: L->E_out, T->S_out, R->W_out
    # S_in: L->W_out, T->N_out, R->E_out
    # E_in: L->S_out, T->W_out, R->N_out
    # W_in: L->N_out, T->E_out, R->S_out
    mapping = {
        "N_in": {"L": "E_out", "T": "S_out", "R": "W_out"},
        "S_in": {"L": "W_out", "T": "N_out", "R": "E_out"},
        "E_in": {"L": "S_out", "T": "W_out", "R": "N_out"},
        "W_in": {"L": "N_out", "T": "E_out", "R": "S_out"},
    }
    if in_edge not in mapping: return None
    for k,v in mapping[in_edge].items():
        if out_edge == v: return k
    return None

# buckets
ns_tr = ['r'] * N  # NS through+right
ns_l  = ['r'] * N  # NS lefts
ew_tr = ['r'] * N  # EW through+right
ew_l  = ['r'] * N  # EW lefts

for i, col in enumerate(links):
    if not col: continue
    inLane, outLane, via = col[0]
    in_e, out_e = edge_of_lane(inLane), edge_of_lane(outLane)
    mt = mov_type(in_e, out_e)
    if in_e in ("N_in", "S_in"):
        if mt in ("T","R"): ns_tr[i] = 'G'
        if mt == "L":       ns_l[i]  = 'G'
    if in_e in ("E_in", "W_in"):
        if mt in ("T","R"): ew_tr[i] = 'G'
        if mt == "L":       ew_l[i]  = 'G'

def s(lst): return "".join(lst)

YEL_NS_TR = ''.join('y' if c=='G' else 'r' for c in ns_tr)
YEL_NS_L  = ''.join('y' if c=='G' else 'r' for c in ns_l)
YEL_EW_TR = ''.join('y' if c=='G' else 'r' for c in ew_tr)
YEL_EW_L  = ''.join('y' if c=='G' else 'r' for c in ew_l)
ALLR      = 'r' * N

# timings (tune if needed)
G_TR   = 35   # straight+right
G_L    = 20   # protected left
YEL    = 4
ALLRED = 2

xml = f"""<additional>
  <tlLogic id="{tid}" type="static" programID="baseline_protected" offset="0">
    <!-- 1) NS through+right -->
    <phase duration="{G_TR}"   state="{s(ns_tr)}"/>
    <phase duration="{YEL}"    state="{YEL_NS_TR}"/>
    <phase duration="{ALLRED}" state="{ALLR}"/>

    <!-- 2) NS left only -->
    <phase duration="{G_L}"    state="{s(ns_l)}"/>
    <phase duration="{YEL}"    state="{YEL_NS_L}"/>
    <phase duration="{ALLRED}" state="{ALLR}"/>

    <!-- 3) EW through+right -->
    <phase duration="{G_TR}"   state="{s(ew_tr)}"/>
    <phase duration="{YEL}"    state="{YEL_EW_TR}"/>
    <phase duration="{ALLRED}" state="{ALLR}"/>

    <!-- 4) EW left only -->
    <phase duration="{G_L}"    state="{s(ew_l)}"/>
    <phase duration="{YEL}"    state="{YEL_EW_L}"/>
    <phase duration="{ALLRED}" state="{ALLR}"/>
  </tlLogic>
</additional>
"""
open("cross.tll.xml","w").write(xml)
print(f"Wrote cross.tll.xml for {tid} with {N} signals.")
traci.close()
