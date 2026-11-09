"""Static prior of the scene (here the lower floor of the house): free space for the base, a height field for the body.

The map is built offline from the scene's collision geometry (dev/build_scene_map.py) and is the
same for every task instance. Grid frame "M" is the scene frame of the map file. It contains glass
partitions and windows, which the depth cameras cannot see.
"""
import heapq
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from scipy.spatial import cKDTree


class SceneMap:
    def __init__(self, path, base_radius=.40):
        data = np.load(path); meta = json.loads(str(data['meta']))
        self.x0, self.y0, self.res = meta['x0'], meta['y0'], meta['res']
        self.shape = tuple(meta['shape']); self.meta = meta; self.base_radius = base_radius
        self.top = data['top'].astype(np.float32); self.desk = data['desk']; self.glass = data['glass']; self.upper = data['upper']
        self.top_static = data['top_static'].astype(np.float32)
        # Furniture the base can stand under (a desk): boxes instead of height-field columns.
        self.boxes = data['boxes'].astype(float) if 'boxes' in data.files else np.zeros((0, 6))
        detail=Path(path).with_name('contact_geometry.npz')
        self.contact_boxes=None;self.contact_convex=[];self.precise_support_hulls=False
        if detail.exists():
            contact=np.load(detail)
            self.contact_boxes=contact['boxes'].astype(float)
            self.contact_fixed=contact['fixed'].astype(bool)
            self.contact_convex=[contact[k].astype(float) for k in contact.files if k.startswith('convex_planes_')]
            self.precise_support_hulls=bool(self.contact_convex)
            self.contact_index={}
            for fixed_only in (False,True):
                boxes=self.contact_boxes[self.contact_fixed] if fixed_only else self.contact_boxes
                middle=(boxes[:,:3]+boxes[:,3:])/2;half=(boxes[:,3:]-boxes[:,:3])/2
                self.contact_index[fixed_only]=(boxes,middle,half,cKDTree(middle))
        support=Path(path).with_name('support_geometry.npz')
        self.support_grid=dict(np.load(support)) if support.exists() else None
        # What the depth camera sees, for localisation (window panes instead of frame hulls, no chairs).
        self.sight_low = data['sight_low'] if 'sight_low' in data.files else None
        self.sight_upper = data['sight_upper'] if 'sight_upper' in data.files else None
        # ... or, where the map has them, the same in thin height slabs (see dev/build_scene_map.py).
        self.sight_slabs = [tuple(s) for s in meta.get('sight_slabs') or []]
        self.sight = [data[f'sight_{k}'] for k in range(len(self.sight_slabs))] if 'sight_0' in data.files else []
        movable = data['movable']
        block = data['low'] | data['mid']
        self.block = block
        # Chairs can be pushed; planning may ignore them when asked to.
        self.block_static = block & ~binary_dilation(movable, iterations=2)
        self.distance = distance_transform_edt(~block) * self.res
        self.distance_static = distance_transform_edt(~self.block_static) * self.res
        self.movable = movable
        # What the chassis itself can touch, fixed and movable apart, for places too narrow for the disc that
        # base_free() takes the chassis for.
        fixed = data['chassis'] if 'chassis' in data.files else block
        self.chassis_distance = distance_transform_edt(~fixed) * self.res
        self.chassis_movable_distance = distance_transform_edt(~(data['chassis_movable'] if 'chassis_movable' in data.files else movable)) * self.res
        self._stencils = {}
        self._di, self._dj, self._d2 = self._stencil(.22)
        # Coarse grid for route planning.
        self.coarse = 2

    def _stencil(self, radius):
        """Cell offsets that cover a disc of @radius around a cell, and their squared distances (m^2)."""
        reach = int(np.ceil(radius / self.res))
        if reach not in self._stencils:
            k = np.arange(-reach, reach + 1); di, dj = np.meshgrid(k, k, indexing='ij')
            self._stencils[reach] = (di, dj, (di**2 + dj**2) * self.res**2)
        return self._stencils[reach]

    def cell(self, xy):
        xy = np.asarray(xy, float)
        return np.floor((xy[..., 0] - self.x0) / self.res).astype(int), np.floor((xy[..., 1] - self.y0) / self.res).astype(int)

    def inside(self, i, j):
        return (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])

    def clearance(self, xy, static_only=False):
        """Distance from @xy to the nearest geometry that blocks the base (metres); 0 outside the map."""
        i, j = self.cell(xy); ok = self.inside(i, j)
        field = self.distance_static if static_only else self.distance
        return np.where(ok, field[np.clip(i, 0, self.shape[0] - 1), np.clip(j, 0, self.shape[1] - 1)], 0.)

    def base_free(self, xy, margin=0., static_only=False):
        return bool(self.clearance(xy, static_only) > self.base_radius + margin)

    def footprint_free(self, x, y, yaw, outline, margin=.03, movable_margin=0.):
        """True if the chassis, a rectangle @outline = (rear, front, half width) in the base frame, standing at
        (@x, @y, @yaw) keeps @margin from fixed things at chassis height and @movable_margin from movable ones
        (a table that is touched gives way; a negative value lets the chassis overlap them by that much).
        Distances are to the middles of the map's cells (0.02 m)."""
        c, s = np.cos(yaw), np.sin(yaw); step = .015

        def cells(shrink):
            rear, front, half = outline[0] + shrink, outline[1] - shrink, outline[2] - shrink
            u, v = np.meshgrid(np.r_[np.arange(rear, front, step), front], np.r_[np.arange(-half, half, step), half], indexing='ij')
            return self.cell(np.c_[x + c * u.ravel() - s * v.ravel(), y + s * u.ravel() + c * v.ravel()])

        i, j = cells(0.)
        if not np.all(self.inside(i, j)) or not np.all(self.chassis_distance[i, j] > margin):
            return False
        i, j = cells(max(0., -movable_margin))
        return bool(np.all(self.chassis_movable_distance[i, j] > max(movable_margin, .001)))

    def sphere_clearance(self, centers, radii, ignore_below=None, static_only=False):
        """Per-sphere gap (m) between each sphere and the height field; negative means penetration.

        static_only: leave out objects that can be pushed (chairs), for places the base has cleared.
        ignore_below: optional per-sphere height; geometry lower than it is disregarded for that
        sphere (used for the fingers, which are meant to work at floor level).
        """
        centers = np.asarray(centers, float); radii = np.asarray(radii, float)
        # The stencil must cover the largest sphere. The robot's own spheres fit the default one;
        # a larger probe (the lean-room test of stance planning) gets a stencil of its own size.
        # Before m05 every sphere used the default: geometry further than 0.22 m from the centre
        # of a large probe was not seen at all.
        di, dj, d2 = (self._di, self._dj, self._d2) if radii.max(initial=0.) <= .22 else self._stencil(float(radii.max()))
        i, j = self.cell(centers[:, :2])
        ii = i[:, None, None] + di[None]; jj = j[:, None, None] + dj[None]
        ok = self.inside(ii, jj)
        field = self.top_static if static_only else self.top
        top = np.where(ok, field[np.clip(ii, 0, self.shape[0] - 1), np.clip(jj, 0, self.shape[1] - 1)], 2.)
        if ignore_below is not None:
            top = np.where(top < np.asarray(ignore_below, float)[:, None, None], -1., top)
        r2 = radii[:, None, None]**2; within = d2[None] < r2
        drop = np.sqrt(np.maximum(r2 - d2[None], 0.))
        gap = np.where(within, centers[:, 2, None, None] - drop - top, np.inf).min(axis=(1, 2))
        if len(self.boxes):
            nearest = np.clip(centers[None], self.boxes[:, None, :3], self.boxes[:, None, 3:])
            gap = np.minimum(gap, (np.linalg.norm(centers[None] - nearest, axis=2) - radii[None]).min(axis=0))
        return gap

    def contact_clearance(self,centers,radii,static_only=False,ignore_below=None):
        """Continuous conservative mesh AABBs for contact, without heightfield raster alias.

        Each box encloses one canonical collision mesh. Signed box distance also rejects
        a sphere centre deep inside geometry; sampling and positive margins remain unchanged.
        """
        if self.contact_boxes is None:
            return self.sphere_clearance(centers,radii,static_only=static_only)
        centers=np.asarray(centers,float);radii=np.asarray(radii,float)
        boxes,middle,half,tree=self.contact_index[bool(static_only)]
        # Exact broad phase. Nearby box centres supply an upper bound on each
        # point's nearest-box distance. A box farther from the entire query
        # AABB than every such upper bound cannot attain the minimum. Keep all
        # intersecting boxes, including any that contain a query point.
        if len(centers) and len(boxes)>32:
            indices=tree.query(centers,k=min(8,len(boxes)))[1]
            seed_delta=abs(centers[:,None,:]-middle[indices])-half[indices]
            seed_distance=np.linalg.norm(np.maximum(seed_delta,0.),axis=2)+np.minimum(np.max(seed_delta,axis=2),0.)
            if ignore_below is not None:
                seed_distance=np.where(boxes[indices,5]<np.asarray(ignore_below)[:,None],np.inf,seed_distance)
            upper=float(np.max(np.min(seed_distance,axis=1)))
            if np.isfinite(upper):
                separation=np.maximum(np.maximum(boxes[:,:3]-centers.max(0),centers.min(0)-boxes[:,3:]),0.)
                possible=np.linalg.norm(separation,axis=1)<=max(0.,upper)+1e-10
                boxes=boxes[possible];middle=middle[possible];half=half[possible]
        delta=abs(centers[:,None,:]-middle)-half
        distance=np.linalg.norm(np.maximum(delta,0.),axis=2)+np.minimum(np.max(delta,axis=2),0.)
        if ignore_below is not None:
            distance=np.where(boxes[None,:,5]<np.asarray(ignore_below)[:,None],np.inf,distance)
        gap=np.min(distance,axis=1)-radii
        for planes in self.contact_convex:
            # Maximum signed face distance is a conservative lower bound outside
            # a convex hull and the exact nearest-face distance inside it.
            gap=np.minimum(gap,np.max(centers@planes[:,:3].T+planes[:,3],axis=1)-radii)
        i,j=self.cell(centers[:,:2])
        return np.where(self.inside(i,j),gap,-2.)

    def support_heights(self,points):
        result=np.full(points.shape[:-1],float(self.meta.get('floor_height',0.)))
        for surface in self.meta['shelf_surfaces']:
            b=np.asarray(surface['bounds']);planes=np.asarray(surface['planes'])
            heights=-(points[...,:2]@planes[:,:2].T+planes[:,3])/planes[:,2]
            height=heights.min(-1)
            inside=np.all(points[...,:2]>=b[:2]-.001,axis=-1)&np.all(points[...,:2]<=b[3:5]+.001,axis=-1)&(height<=points[...,2]+.004)
            result=np.where(inside,np.maximum(result,height),result)
        return result

    def path(self, start, goal, margin=.02, static_only=False, max_expansions=60000, keepout=()):
        """A* over base positions; returns a list of xy waypoints (simplified) or None.

        keepout: (xy, radius) discs the base centre must stay out of (objects that are not in the map).
        """
        c = self.coarse; res = self.res * c
        field = (self.distance_static if static_only else self.distance)[::c, ::c]
        if keepout:
            field = field.copy(); gx = self.x0 + (np.arange(field.shape[0]) + .5) * res; gy = self.y0 + (np.arange(field.shape[1]) + .5) * res
            for centre, radius in keepout:
                away = np.hypot(gx[:, None] - centre[0], gy[None, :] - centre[1]) - radius
                # Express the disc as "distance to an obstacle" so the usual clearance test applies.
                field = np.minimum(field, away + self.base_radius + margin)
        free = field > self.base_radius + margin
        def to_cell(xy):
            return (int((xy[0] - self.x0) / res), int((xy[1] - self.y0) / res))
        def to_xy(cell):
            return np.array([self.x0 + (cell[0] + .5) * res, self.y0 + (cell[1] + .5) * res])
        a, b = to_cell(start), to_cell(goal); n, m = free.shape
        if not (0 <= b[0] < n and 0 <= b[1] < m) or not (0 <= a[0] < n and 0 <= a[1] < m):
            return None
        free = free.copy()
        # The robot may already stand closer to something than the planning margin: let it leave.
        # At a work surface it stands with the chassis against (or under) the furniture: the start may be
        # much closer than the goal.
        for cell, slack in ((a, .20), (b, .10)):
            if not free[cell]:
                xs, ys = np.ogrid[:n, :m]; near = (xs - cell[0])**2 + (ys - cell[1])**2 <= ((slack + .02) / res)**2
                free |= near & (field > self.base_radius - slack)
        if not free[b] or not free[a]:
            return None
        queue = [(0., 0., a)]; cost = {a: 0.}; previous = {}; expanded = 0
        while queue and expanded < max_expansions:
            expanded += 1
            _, g, u = heapq.heappop(queue)
            if g > cost[u] + 1e-9:
                continue
            if u == b:
                cells = [u]
                while u != a:
                    u = previous[u]; cells.append(u)
                points = [np.asarray(start, float)[:2]] + [to_xy(x) for x in cells[::-1][1:-1]] + [np.asarray(goal, float)[:2]]
                return self.simplify(points, margin, static_only, keepout)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < n and 0 <= v[1] < m) or not free[v]:
                    continue
                if dx and dy and not (free[u[0] + dx, u[1]] and free[u[0], u[1] + dy]):
                    continue
                # Prefer the middle of corridors: small penalty when close to obstacles.
                step = np.hypot(dx, dy) * res * (1. + .6 * max(0., .25 - (field[v] - self.base_radius)) / .25)
                ng = g + step
                if ng < cost.get(v, np.inf):
                    cost[v] = ng; previous[v] = u
                    heapq.heappush(queue, (ng + np.hypot(v[0] - b[0], v[1] - b[1]) * res, ng, v))
        return None

    def chassis_path(self,start,goal,yaw,outline=(-.40,.24,.345),max_expansions=20000,valid=None):
        """Fixed-heading rectangular chassis path; caller must check full body and movable objects.

        Used only where the existing circular route cannot approach a work surface.
        Endpoints and continuous legs retain the same footprint margins as surface stances.
        """
        start=np.asarray(start,float);goal=np.asarray(goal,float)
        def segment(a,b):
            count=max(2,int(np.ceil(np.linalg.norm(b-a)/.01))+1)
            return all(self.footprint_free(*point,yaw,outline,margin=.015,movable_margin=.02)
                       or (valid is not None and valid(point)) for point in np.linspace(a,b,count))
        if not segment(start,start) or not segment(goal,goal):return None
        reach=int(np.ceil((max(abs(outline[0]),abs(outline[1]))+outline[2])/self.res))+1
        offsets=np.arange(-reach,reach+1)*self.res
        x,y=np.meshgrid(offsets,offsets,indexing='ij');c,s=np.cos(yaw),np.sin(yaw)
        u=c*x+s*y;v=-s*x+c*y
        stencil=(u>=outline[0])&(u<=outline[1])&(abs(v)<=outline[2])
        free=binary_erosion((self.chassis_distance>.015)&(self.chassis_movable_distance>.02),structure=stencil)
        def cell(point):return tuple(int(v) for v in self.cell(point))
        def xy(node):return np.array([self.x0+(node[0]+.5)*self.res,self.y0+(node[1]+.5)*self.res])
        origin,destination=cell(start),cell(goal)
        if not self.inside(*origin) or not self.inside(*destination):return None
        # Validate continuous endpoint connectors. Erosion offsets and the sampled footprint
        # can differ by one raster cell; never force an untested endpoint through geometry.
        for endpoint in (start,goal):
            node=cell(endpoint)
            for di in range(-2,3):
                for dj in range(-2,3):
                    nearby=(node[0]+di,node[1]+dj)
                    if self.inside(*nearby) and segment(endpoint,xy(nearby)):
                        free[nearby]=True
        if not free[origin] or not free[destination]:return None
        queue=[(float(np.linalg.norm(start-goal)),0.,origin)];cost={origin:0.};parent={};expanded=0
        while queue and expanded<max_expansions:
            _,g,node=heapq.heappop(queue)
            if g>cost[node]+1e-9:continue
            expanded+=1
            if node==destination:
                nodes=[destination]
                while nodes[-1]!=origin:nodes.append(parent[nodes[-1]])
                points=[start]+[xy(n) for n in nodes[::-1][1:-1]]+[goal]
                result=[points[0]];i=0
                while i<len(points)-1:
                    j=len(points)-1
                    while j>i+1 and not segment(points[i],points[j]):j-=1
                    if not segment(points[i],points[j]):return None
                    result.append(points[j]);i=j
                return result
            for di,dj in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                nxt=(node[0]+di,node[1]+dj)
                if not self.inside(*nxt) or not free[nxt]:continue
                if di and dj and not (free[node[0]+di,node[1]] and free[node[0],node[1]+dj]):continue
                new=g+np.hypot(di,dj)*self.res
                if new>=cost.get(nxt,np.inf):continue
                cost[nxt]=new;parent[nxt]=node
                heapq.heappush(queue,(new+float(np.linalg.norm(xy(nxt)-goal)),new,nxt))
        return None

    def segment_free(self, a, b, margin=.02, static_only=False, keepout=()):
        a = np.asarray(a, float); b = np.asarray(b, float)
        count = max(2, int(np.linalg.norm(b - a) / (self.res * 1.5)) + 1)
        points = a[None] + np.linspace(0, 1, count)[:, None] * (b - a)[None]
        if not np.all(self.clearance(points, static_only) > self.base_radius + margin - .03):
            return False
        return all(np.all(np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) > radius - .03) for centre, radius in keepout)

    def simplify(self, points, margin=.02, static_only=False, keepout=()):
        out = [points[0]]; i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.segment_free(points[i], points[j], margin, static_only, keepout):
                j -= 1
            out.append(points[j]); i = j
        return out
