import numpy as np


# returns a set of tuples { (x,y,z,mip) } for a given mip level
def get_mip_level(dump, lvl):
    mip_size = 128 * 128 * 128
    raw_mip_data = dump[mip_size*lvl : mip_size*lvl+mip_size]
    mip_3d_data = np.reshape(raw_mip_data, (128, 128, 128), order='F')
    return mip_3d_data

# returns a set of tuples { (x,y,z,mip) } for a given mip level
def get_density_points(mip_level, lvl):
    density_points = np.transpose(np.nonzero(mip_level))
    return set(tuple([*xyz, lvl]) for xyz in density_points)

class NgpGrid:
    def __init__(self, path):
        dump = np.fromfile(path, dtype=np.byte)
        # density_points is a set of tuples { (x,y,z,mip) }: constant access & remove
        self.density_points = set()
        self.mip_levels = []
        # add points of all mip levels
        for lvl in range(8):
            mip_level = get_mip_level(dump, lvl)
            mip_points = get_density_points(mip_level, lvl)
            self.mip_levels.append(mip_level)
            # filter out points that are included in lower mip levels
            if lvl > 0:
                mip_points = {(x,y,z,mip) for (x,y,z,mip) in mip_points if not (32<=x<96 and 32<=y<96 and 32<=z<96)}
            self.density_points.update(mip_points)
            print("mip", lvl, len(mip_points))
        print ("total voxels", len(self.density_points))
        
    def get_neighbors(self, point):
        """
        Returns immediate neighbors as list of point tuples [(x,y,z,mip)]
        Args:
            point: tuple (x,y,z,mip)
        """
        x,y,z,mip = point 
        neighbors = []
        
        # check left and right 
        if (x-1, y, z, mip) in self.density_points: neighbors.append((x-1, y, z, mip))
        if (x+1, y, z, mip) in self.density_points: neighbors.append((x+1, y, z, mip))
        # check top and bottom neighbors
        if (x, y-1, z, mip) in self.density_points: neighbors.append((x, y-1, z, mip))
        if (x, y+1, z, mip) in self.density_points: neighbors.append((x, y+1, z, mip))
        # check front and back neighbors
        if (x, y, z-1, mip) in self.density_points: neighbors.append((x, y, z-1, mip))
        if (x, y, z+1, mip) in self.density_points: neighbors.append((x, y, z+1, mip))
        
        # find neighbor at child->parent boundary
        if mip < 7:
            # indices in parent mip
            mx = 32 + x // 2 
            my = 32 + y // 2
            mz = 32 + z // 2
            if x == 0 and (31, my, mz, mip+1) in self.density_points:
                neighbors.append((31, my, mz, mip+1))
            if x == 127 and (96, my, mz, mip+1) in self.density_points:
                neighbors.append((96, my, mz, mip+1))
            if y == 0 and (mx, 31, mz, mip+1) in self.density_points:
                neighbors.append((mx, 31, mz, mip+1))
            if y == 127 and (mx, 96, mz, mip+1) in self.density_points:
                neighbors.append((mx, 96, mz, mip+1))
            if z == 0 and (mx, my, 31, mip+1) in self.density_points:
                neighbors.append((mx, my, 31, mip+1))
            if z == 127 and (mx, my, 96, mip+1) in self.density_points:
                neighbors.append((mx, my, 96, mip+1))
        
        # find neighbor at parent->child boundary
        if mip > 0:
            # indices in child mip
            cx = x * 2 - 64
            cy = y * 2 - 64
            cz = z * 2 - 64
            if x == 31:
                if (0, cy+0, cz+0, mip-1) in self.density_points: neighbors.append((0, cy+0, cz+0, mip-1))
                if (0, cy+0, cz+1, mip-1) in self.density_points: neighbors.append((0, cy+0, cz+1, mip-1))
                if (0, cy+1, cz+0, mip-1) in self.density_points: neighbors.append((0, cy+1, cz+0, mip-1))
                if (0, cy+1, cz+1, mip-1) in self.density_points: neighbors.append((0, cy+1, cz+1, mip-1))
            if x == 96:
                if (127, cy+0, cz+0, mip-1) in self.density_points: neighbors.append((127, cy+0, cz+0, mip-1))
                if (127, cy+0, cz+1, mip-1) in self.density_points: neighbors.append((127, cy+0, cz+1, mip-1))
                if (127, cy+1, cz+0, mip-1) in self.density_points: neighbors.append((127, cy+1, cz+0, mip-1))
                if (127, cy+1, cz+1, mip-1) in self.density_points: neighbors.append((127, cy+1, cz+1, mip-1))
            if y == 31:
                if (cx+0, 0, cz+0, mip-1) in self.density_points: neighbors.append((cx+0, 0, cz+0, mip-1))
                if (cx+0, 0, cz+1, mip-1) in self.density_points: neighbors.append((cx+0, 0, cz+1, mip-1))
                if (cx+1, 0, cz+0, mip-1) in self.density_points: neighbors.append((cx+1, 0, cz+0, mip-1))
                if (cx+1, 0, cz+1, mip-1) in self.density_points: neighbors.append((cx+1, 0, cz+1, mip-1))
            if y == 96:
                if (cx+0, 127, cz+0, mip-1) in self.density_points: neighbors.append((cx+0, 127, cz+0, mip-1))
                if (cx+0, 127, cz+1, mip-1) in self.density_points: neighbors.append((cx+0, 127, cz+1, mip-1))
                if (cx+1, 127, cz+0, mip-1) in self.density_points: neighbors.append((cx+1, 127, cz+0, mip-1))
                if (cx+1, 127, cz+1, mip-1) in self.density_points: neighbors.append((cx+1, 127, cz+1, mip-1))
            if z == 31:
                if (cx+0, cy+0, 0, mip-1) in self.density_points: neighbors.append((cx+0, cy+0, 0, mip-1))
                if (cx+0, cy+1, 0, mip-1) in self.density_points: neighbors.append((cx+0, cy+1, 0, mip-1))
                if (cx+1, cy+0, 0, mip-1) in self.density_points: neighbors.append((cx+1, cy+0, 0, mip-1))
                if (cx+1, cy+1, 0, mip-1) in self.density_points: neighbors.append((cx+1, cy+1, 0, mip-1))
            if z == 96:
                if (cx+0, cy+0, 127, mip-1) in self.density_points: neighbors.append((cx+0, cy+0, 127, mip-1))
                if (cx+0, cy+1, 127, mip-1) in self.density_points: neighbors.append((cx+0, cy+1, 127, mip-1))
                if (cx+1, cy+0, 127, mip-1) in self.density_points: neighbors.append((cx+1, cy+0, 127, mip-1))
                if (cx+1, cy+1, 127, mip-1) in self.density_points: neighbors.append((cx+1, cy+1, 127, mip-1))
                
        return neighbors

    def cluster(self):
        clusters = []
        noise = []
        unvisited_points = self.density_points.copy()
        while len(unvisited_points) > 0:
            P = unvisited_points.pop()
            neighbors = self.get_neighbors(P)
            if len(neighbors) > 0:
                C = {P} # next cluster
                # expand cluster
                i = 0
                while i < len(neighbors):
                    P = neighbors[i]
                    if P in unvisited_points: # if P' is not visited
                        unvisited_points.remove(P) # mark as visited
                        N = self.get_neighbors(P)
                        if len(N) > 0:
                            neighbors.extend(N)
                    C.add(P)
                    i += 1
                clusters.append(C)
            else:
                noise.append(P)
        return clusters, noise

    def serialized(self):
        buf = np.zeros((128, 128, 128, 8), dtype=np.byte)
        for point in self.density_points:
            x,y,z,mip = point
            buf[x,y,z,mip] = 1
            for lvl in range(mip+1, 8):
                x = 32 + x // 2
                y = 32 + y // 2
                z = 32 + z // 2
                buf[x,y,z,lvl] = 1
        data = buf.reshape(128*128*128*8, order='F')
        return data.tobytes()

    # @staticmethod
    # def serialize_data(path, point_set):
    #     buf = np.zeros((128, 128, 128, 8), dtype=np.byte)
    #     for point in point_set:
    #         x,y,z,mip = point
    #         buf[x,y,z,mip] = 1
    #         for lvl in range(mip+1, 8):
    #             x = 32 + x // 2
    #             y = 32 + y // 2
    #             z = 32 + z // 2
    #             buf[x,y,z,lvl] = 1
    #     data = buf.reshape(128*128*128*8, order='F')
    #     with open(path, "wb") as file:
    #         file.write(data.tobytes())
