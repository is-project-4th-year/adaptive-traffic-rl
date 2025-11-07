# ~/traffic_rl/rl_common/env.py
import os
import time
import numpy as np
import traci

# If you're launching SUMO yourself, this flag would choose attach vs spawn.
# In our live setup we ATTACH to the controller's SUMO at :8813.
ATTACH_ONLY = os.getenv("SUMO_ATTACH_ONLY", "1") == "1"  # default attach

class SumoEnvironment:
    def __init__(self, net_file, rou_file, add_files, det_ids,
                 tls_id=None, gui=False, step_length=1.0):
        self.net_file = net_file
        self.rou_file = rou_file
        self.add_files = add_files
        self.det_ids = det_ids
        self.tls_id = tls_id
        self.gui = gui
        self.step_length = step_length
        self.sumo_bin = "sumo-gui" if gui else "sumo"

    def _cmd(self):
        """
        Only used if you ever want to LAUNCH SUMO from this env rather than attach.
        Not used in our live attach flow, but kept for completeness.
        """
        cmd = [
            self.sumo_bin, "-n", self.net_file, "-r", self.rou_file,
            "--step-length", str(self.step_length),
            "--start", "true", "--no-step-log", "true", "--quit-on-end", "true",
            "--additional-files", ",".join(self.add_files),
            "--num-clients", "3",
        ]
        if os.getenv("SUMO_TRIPINFO"):
            cmd += ["--tripinfo-output", os.getenv("SUMO_TRIPINFO")]
        if os.getenv("SUMO_EMISSIONS"):
            cmd += ["--emission-output", os.getenv("SUMO_EMISSIONS")]
        return cmd

    def reset(self):
        """
        Live mode: attach to the controller's already-running SUMO on port 8813.
        Make attach rock-solid:
          - close any previous TraCI
          - retry connect
          - setOrder(3) as FIRST command after connect
          - sanity ping (getVersion) to confirm the wire is stable
        """
        if traci.isLoaded():
            try:
                traci.close()
            except Exception:
                pass

        max_tries = 40
        last_err = None
        for _ in range(max_tries):
            try:
                conn = traci.connect(port=8813)    # controller=1, collector=2, rl-live=3
                # MUST be the first command after connect, or SUMO may drop us
                conn.setOrder(3)

                # Sanity check: forces a round-trip on the channel
                _ = traci.getVersion()
                last_err = None
                break
            except Exception as e:
                last_err = e
                # close half-open session if any
                try:
                    if traci.isLoaded():
                        traci.close()
                except Exception:
                    pass
                time.sleep(0.75)
        else:
            raise RuntimeError(f"rl-live: failed to attach to SUMO on :8813 ({last_err})")

        if self.tls_id is None:
            ids = traci.trafficlight.getIDList()
            self.tls_id = ids[0] if ids else None

        return self.get_state()

    def get_state(self):
        """
        Returns an 8-dim vector:
          [N_cnt, S_cnt, E_cnt, W_cnt, N_spd, S_spd, E_spd, W_spd]
        (Counts use getLastStepVehicleNumber; speeds NaN→0.0)
        """
        cnt, spd = [], []
        for d in self.det_ids:
            cnt.append(traci.inductionloop.getLastStepVehicleNumber(d))
            v = traci.inductionloop.getLastStepMeanSpeed(d)
            spd.append(0.0 if np.isnan(v) else v)
        return np.array(cnt + spd, dtype=np.float32)

    def get_reward(self):
        """
        Simple negative-queue reward (lower is better).
        """
        return -float(sum(traci.inductionloop.getLastStepVehicleNumber(d) for d in self.det_ids))

    def _phase_count(self):
        """
        Count phases for the current TLS program.
        """
        if not self.tls_id:
            return 0
        defs = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)
        try:
            cur = traci.trafficlight.getProgram(self.tls_id)
        except Exception:
            cur = None
        for lgc in defs:
            if getattr(lgc, "programID", None) == cur:
                return len(lgc.phases)
        return len(defs[0].phases) if defs else 0

    def step(self, action):
        """
        Observer mode by default: do NOT fight the controller.
        If you explicitly unset RL_OBSERVER (set it to "0"), action=1 will switch phase.
        We still always tick the simulation by 5s (in 1s steps) to keep state fresh.
        """
        observer = os.getenv("RL_OBSERVER", "1") == "1"
        if not observer and self.tls_id and action == 1:
            cur = traci.trafficlight.getPhase(self.tls_id)
            m = self._phase_count()
            if m > 0:
                traci.trafficlight.setPhase(self.tls_id, (cur + 1) % m)

        # Advance 5s worth of sim time
        steps = max(1, int(round(5.0 / float(self.step_length))))
        for _ in range(steps):
            traci.simulationStep()

        s = self.get_state()
        r = self.get_reward()
        done = (traci.simulation.getMinExpectedNumber() == 0 and traci.simulation.getLoadedNumber() == 0)
        return s, r, done, {}
