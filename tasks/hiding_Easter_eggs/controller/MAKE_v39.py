"""eggs v39 = frozen v38 + explore toward a parked candidate with strong evidence (line endings kept; each edit once)."""
import sys
p = sys.argv[1] + "/easter_policy.py"
s = open(p, newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
L = lambda *xs: nl.join(xs)
edits = [
    ("from occupancy import GardenMap, rot2, wrap", "from occupancy import COARSE, N, RES, GardenMap, rot2, wrap"),
    (L("    def update_candidates(self, head):"),
     L("    def toward_cell(self, xy):                                                                # reachable planning cell nearest to xy, if >= 1 m nearer than the robot",
       "        travel = self.map.travel_costs(self.pose[:2])",
       "        ii, jj = np.nonzero(np.isfinite(travel))",
       "        if len(ii) == 0:",
       "            return None",
       "        cells = (np.c_[ii, jj] * COARSE + COARSE / 2 - N / 2) * RES                             # [k, 2] odometry frame",
       "        d = np.linalg.norm(cells - xy, axis=1)",
       "        k = int(np.argmin(d))",
       "        return cells[k] if d[k] < np.linalg.norm(self.pose[:2] - xy) - 1.0 else None",
       "",
       "    def update_candidates(self, head):")),
    (L("                if self.path is None or self.step - self.planned > 90:",
       "                    self.goal, gain = self.map.viewpoint(self.pose[:2], self.visited, keep=self.goal)"),
     L("                if self.path is None or self.step - self.planned > 90:",
       "                    strong = [c for c in self.candidates if c.get(\"parked\") and not c[\"dead\"] and not c.get(\"missed\") and c[\"hits\"] >= 20 and c[\"misses\"] <= 2]",
       "                    lure = self.toward_cell(max(strong, key=lambda c: c[\"hits\"])[\"xy\"]) if strong else None   # v3.9.3 305: the real basket parked at 91 hits / 0 misses,",
       "                    if lure is not None:                                                  # unplannable from 10 m; frontiers elsewhere took the rest of the run",
       "                        self.goal, gain = lure, -2.0",
       "                        self.log[\"lure\"] = self.log.get(\"lure\", 0) + 1",
       "                    else:",
       "                        self.goal, gain = self.map.viewpoint(self.pose[:2], self.visited, keep=self.goal)")),
]
for a, b in edits:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w", newline="").write(s)
print("eggs v39 written", len(edits))
