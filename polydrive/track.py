"""Centerline for Summer 1, read off the track's own part layout (grid units).

Grid -> world is x5; road surface sits about 0.35 above y*5. Progress is the
arc length of the car's projection onto this polyline, searched only in a
window around the last known progress so parallel or crossing roads (the
return road, the jump over the first straight) can't be confused.
"""
import numpy as np

# (x, y, z) in grid units, in driving order.
SUMMER1 = [
    (64, 11, 4), (56, 11, 1), (48, 11, 0), (36, 11, 0),   # start -> CP0
    (30, 10, 0), (24, 7, 0), (16, 4, 0), (8, 0, -1),      # downhill
    (0, 0, -2), (-20, 0, -2), (-36, 0, -2),               # wide straight -> CP1
    (-44, 1, -4), (-50, 3, -10), (-51, 3, -18), (-48, 3, -24),  # banked turn
    (-40, 2, -24), (-26, 2, -24), (-20, 2, -20),          # return road
    (-20, 2, -16), (-20, 2, -8),                          # CP2 -> ramp
    (-20, 4, 4), (-20, 4, 14),                            # jump, landing
    (-20, 4, 30), (-20, 4, 46), (-28, 4, 52),             # wide section
    (-40, 4, 54), (-52, 4, 56), (-60, 4, 56),             # -> finish
]


class Centerline:
    def __init__(self, grid_points=SUMMER1):
        p = np.array(grid_points, dtype=np.float64) * 5.0
        p[:, 1] += 0.35
        self.p = p
        seg = p[1:] - p[:-1]
        self.seg = seg
        self.len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.len)])
        self.total = float(self.cum[-1])

    def project(self, pos, last_progress, back=20.0, ahead=60.0):
        """Return (progress, horizontal distance from line, height above line)."""
        pos = np.asarray(pos, dtype=np.float64)
        best = None
        for i in range(len(self.seg)):
            if self.cum[i + 1] < last_progress - back or self.cum[i] > last_progress + ahead:
                continue
            a, d = self.p[i], self.seg[i]
            # Project in the horizontal plane; height is checked separately.
            dh = d[[0, 2]]
            t = np.clip(np.dot((pos - a)[[0, 2]], dh) / max(np.dot(dh, dh), 1e-9), 0.0, 1.0)
            q = a + t * d
            lateral = float(np.linalg.norm((pos - q)[[0, 2]]))
            if best is None or lateral < best[1]:
                best = (float(self.cum[i] + t * self.len[i]), lateral, float(pos[1] - q[1]))
        return best
