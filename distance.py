"""This program reads a DataFrame from a CSV file and evaluates different distances between the track and the modules of the detector."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from channel_mapping import CAT_MODULES, MEM_MODULES, MODULES, GetValidModules, module_mapping

DATAFRAME = "data/processed_dataframe.csv"
ARAPUCA_DIMENSIONS = (0.05, 0.6, 0.6)
MODULES_COORD = module_mapping()
N = 100

def track_coordinates(df):
    """Extract the start and end coordinates of the track from the DataFrame."""
    start = df[['trkstartx', 'trkstarty', 'trkstartz']].to_numpy()
    end   = df[['trkendx', 'trkendy', 'trkendz']].to_numpy()
    return start, end

def radial_distance(df):
    """Calculate the radial distance as the area of a parallelogram defined by the track and the module, divided by the length of the track."""
    track_start, track_end = track_coordinates(df)
    for m in MODULES:
        x, y, z = MODULES_COORD[m]
        point = np.array([x, y, z])

        d = track_end - track_start                            
        w = point - track_start               
        cross = np.cross(d, w)
        dist_rad = np.linalg.norm(cross, axis=1) / np.linalg.norm(d, axis=1)

        df[f"rad_d_{m}"] = dist_rad
    
    return df

def vertex_position():
    """Calculate the vertexes position of the module, knowing its center coordinates and the module dimensions."""
    R = {}
    for m in MODULES:
        x, y, z = MODULES_COORD[m]
        dx, dy, dz = ARAPUCA_DIMENSIONS
        
        if m in CAT_MODULES:
            r1 = (x, y + dy/2, z - dz/2)
            r2 = (x, y + dy/2, z + dz/2)
            r3 = (x, y - dy/2, z + dz/2)
        
        else: 
            r1 = (x - dx/2, y, z - dz/2)
            r2 = (x + dx/2, y, z - dz/2)
            r3 = (x + dx/2, y, z + dz/2)
        
        R[m] = (r1, r2, r3)

    return R

def track_segments(s, e):
    """Divide the track in N segment and calculate each center"""
    P = []
    for a in range(len(s)):
        d = e[a] - s[a]
        points = []
        for i in range(N):
            t = (i + 0.5) / N
            points.append(s[a] + t*d)
        P.append(points)  
    
    return P

def lenght_segment(s, e): 
    l = []
    for a in range(len(s)):
        d = np.linalg.norm(e[a] - s[a])
        l.append(d/N)
    return l

def integrated_angular_acceptance(df):
    """Calculate the integrated angular acceptance of the track with respect to the module."""
    start, end = track_coordinates(df)
    P = track_segments(start, end)
    R = vertex_position()
    l = lenght_segment(start, end)
    angles = {m:[] for m in MODULES}
    for i in range(len(start)):
        for m, vert in R.items():
            r1, r2, r3 = [np.array(v) for v in vert] 
            total_angle = 0
            for j in range(N):
                v1 = (r1 - P[i][j])/np.linalg.norm(r1 - P[i][j])
                v2 = (r2 - P[i][j])/np.linalg.norm(r2 - P[i][j])
                v3 = (r3 - P[i][j])/np.linalg.norm(r3 - P[i][j])

                cross = np.cross(v2, v3)
                num = np.dot(v1, cross)
                den = 1 + np.dot(v1, v2) + np.dot(v2, v3) + np.dot(v3, v1)
                solid_angle = 2 * np.arctan2(num, den) #to keep the sign information
                solid_angle *= l[i] 
                total_angle += solid_angle 
            angles[m].append(total_angle)

    df = df.assign(**{f"solid_angle_{m}": angles[m] for m in angles})
    
    return df

def main():
    df = pd.read_csv(DATAFRAME)
    df = radial_distance(df)
    df = integrated_angular_acceptance(df)
    df.to_csv("data/processed_dataframe_with_distances.csv", index=False)

if __name__ == "__main__":
    main()


