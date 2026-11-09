"""Bounded fixed-heading A* with continuous robot geometry on every edge."""
import heapq,time
import numpy as np

def path(start,goal,valid,resolution=.03,seconds=7.,max_expansions=5000):
    start=np.asarray(start,float);goal=np.asarray(goal,float);deadline=time.monotonic()+seconds
    cache={}
    def clear(point):
        key=tuple(np.round(point,7))
        if key not in cache:cache[key]=bool(valid(point))
        return cache[key]
    def edge(a,b):
        return all(clear(p) for p in np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.01))+1)))
    if not clear(start) or not clear(goal):return None
    if edge(start,goal):return [start,goal]
    lo=np.minimum(start,goal)-.65;hi=np.maximum(start,goal)+.65
    xy=lambda node:start+resolution*np.array(node)
    origin=(0,0);queue=[(float(np.linalg.norm(goal-start)),0.,origin)];cost={origin:0.};parent={};expanded=0
    while queue and expanded<max_expansions and time.monotonic()<deadline:
        _,g,node=heapq.heappop(queue)
        if g>cost[node]+1e-9:continue
        point=xy(node);expanded+=1
        if np.linalg.norm(point-goal)<resolution*2.1 and edge(point,goal):
            nodes=[node]
            while nodes[-1]!=origin:nodes.append(parent[nodes[-1]])
            route=[xy(n) for n in nodes[::-1]]+[goal];result=[route[0]];i=0
            while i<len(route)-1:
                j=len(route)-1
                while j>i+1 and not edge(route[i],route[j]):j-=1
                result.append(route[j]);i=j
            return result
        for di,dj in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            nxt=(node[0]+di,node[1]+dj);target=xy(nxt)
            if np.any(target<lo) or np.any(target>hi):continue
            ng=g+np.hypot(di,dj)*resolution
            if ng>=cost.get(nxt,np.inf):continue
            if not edge(point,target):continue
            cost[nxt]=ng;parent[nxt]=node
            heapq.heappush(queue,(ng+float(np.linalg.norm(target-goal)),ng,nxt))
    return None
