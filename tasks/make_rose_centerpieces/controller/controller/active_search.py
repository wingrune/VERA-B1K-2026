"""Finite viewpoint changes chosen from observed depth, never scene metadata."""
import numpy as np
from scipy.ndimage import uniform_filter
from navigation import rot2, wrap


class ActiveSearch:
    def __init__(self,global_frontiers=False):
        self.global_frontiers=global_frontiers
        self.visited = []
        self.goal = None
        self.path = None
        self.age = 0
        self.return_phase = None
        self.debug = {}

    def plan(self, depth, odom, phase):
        known = (depth.log_odds[:, :, :8] < 0).any(axis=2)
        occupied = depth.occupancy()
        information = uniform_filter((~known).astype(float), size=35)
        proposals = []
        if self.global_frontiers:
            from frontier_search import frontier_proposals
            proposals=frontier_proposals(depth,odom,self.visited)
        frontier_count=len(proposals)
        radial=[]
        for radius in [.75, 1.2, 1.7, 2.2]:
            for angle in np.linspace(-np.pi, np.pi, 24, endpoint=False):
                goal = odom[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
                cell = depth.ij(goal)
                if not np.all((cell >= 2) & (cell < depth.n - 2)):
                    continue
                if occupied[tuple(cell)] or not known[tuple(cell)]:
                    continue
                novelty = min([np.linalg.norm(goal-p) for p in self.visited] or [2.])
                if novelty < .4:
                    continue
                score = 1.5 * min(novelty, 1.5) + information[tuple(cell)] + .05 * radius - .05 * abs(wrap(angle-odom[2]))
                radial.append((score, goal))
        ranked=sorted(proposals,key=lambda x:-x[0])[:12]+sorted(radial,key=lambda x:-x[0])[:8 if frontier_count else 15]
        for score, goal in ranked:
            path = depth.path(odom[:2], goal,occupancy=occupied)
            if not path:
                continue
            cells = depth.ij(np.asarray(path))
            observed = known[cells[:, 0], cells[:, 1]]
            near = np.linalg.norm(np.asarray(path)-odom[:2], axis=1) < depth.radius + .1
            if np.mean(observed | near) < .8:
                continue
            self.visited.append(odom[:2].copy())
            self.goal, self.path, self.age, self.return_phase = goal, path, 0, phase
            self.motion_budget=min(900 if frontier_count else 480,max(180,int(np.linalg.norm(goal-odom[:2])/.18*30)+60))
            self.debug = dict(goal=goal.tolist(), score=float(score), observed_fraction=float(np.mean(observed | near)), visited=len(self.visited),global_frontier_candidates=frontier_count)
            return True
        self.debug = dict(no_reachable_viewpoint=True, candidates=len(proposals)+len(radial))
        return False

    def step(self, depth, odom, action):
        self.age += 1
        action[:3] = 0
        if np.linalg.norm(self.goal-odom[:2]) < .12 or self.age > self.motion_budget:
            self.visited.append(odom[:2].copy())
            return True
        if self.age % 30 == 1:
            self.path = depth.path(odom[:2], self.goal)
        if not self.path:
            self.visited.append(self.goal.copy())
            return True
        near = np.argmin([np.linalg.norm(x-odom[:2]) for x in self.path])
        waypoint = self.path[min(near+1, len(self.path)-1)]
        for offset in [6,5,4,3,2]:
            candidate=self.path[min(near+offset,len(self.path)-1)]
            if depth.line_clear(odom[:2],candidate):waypoint=candidate;break
        delta = rot2(odom[2]).T @ (waypoint-odom[:2])
        heading = np.arctan2(self.goal[1]-odom[1], self.goal[0]-odom[0])
        action[2] = np.clip(wrap(heading-odom[2]) * .7, -.35, .35)
        if depth.line_clear(odom[:2], waypoint):
            action[:2] = np.clip(delta * .9, -.32, .32)
        return False
