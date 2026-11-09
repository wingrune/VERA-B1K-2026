"""eggs v40 = v39 + an egg view that never had the basket in sight moves the base and looks again (line endings kept; each edit once)."""
import sys
p = sys.argv[1] + "/easter_policy.py"
s = open(p, newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
L = lambda *xs: nl.join(xs)
edits = [
    (L("        self.ring = False                                                                     # approach leg to a point 2.5 m from a far candidate"),
     L("        self.ring = False                                                                     # approach leg to a point 2.5 m from a far candidate",
       "        self.reposition, self.view_seen = 0, False                                            # egg views that never saw the basket: base moves so far")),
    ("        self.path, self.planned, self.turned, self.stuck, self.ring = None, -1000, 0.0, 0, False",
     "        self.path, self.planned, self.turned, self.stuck, self.ring, self.view_seen = None, -1000, 0.0, 0, False, False"),
    (L("                rim = basket_rim(cams[\"right_wrist\"], centre)                                 # the basket may have been dragged: refit its rim from above (rim_refit_test: ~1 cm on"),
     L("                self.view_seen = True",
       "                rim = basket_rim(cams[\"right_wrist\"], centre)                                 # the basket may have been dragged: refit its rim from above (rim_refit_test: ~1 cm on")),
    (L("            elif self.age > 200:                                                              # nothing visible in the basket",
       "                self.views = []",
       "                self.log[\"view_empty\"] = self.step",
       "                self.change(\"done\")"),
     L("            elif self.age > 200:                                                              # nothing visible in the basket",
       "                self.views = []",
       "                if not self.view_seen and self.reposition < 2:                                # the basket never came into view (final v39 301: 1.2 m off the hand): move the base",
       "                    self.reposition += 1",
       "                    self.log[\"reposition\"] = self.reposition",
       "                    self.change(\"go_basket\")",
       "                else:",
       "                    self.log[\"view_empty\"] = self.step",
       "                    self.change(\"done\")")),
]
for a, b in edits:
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
open(p, "w", newline="").write(s)
print("eggs v40 written", len(edits))
