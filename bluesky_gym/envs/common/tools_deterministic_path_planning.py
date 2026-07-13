# -*- coding: utf-8 -*-
"""
Tools module

Created on Tue Jun 14 11:39:20 2016

Adapted from: Daphne Rein-Weston
"""
from math import sqrt,atan2,acos,pi, isnan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from bluesky_gym.envs.common.wind_field_deterministic_path_planning import Windfield

debugging_printing_flag = 0

# enable logging on console
console_logging_flag   = 0
if debugging_printing_flag:
    console_logging_flag   = 1

# Define windfield

#################################################################
def specifywindfield(ymin,ymax,xmin,xmax):
    wind = Windfield()

    # wind.addpoint(lat,lon,winddir,windspd)
    if console_logging_flag:
        print('specify windfield', ymin,ymax,xmin,xmax)
      
    wind.addpoint(ymax,xmin,90.,150.) # deg, kts
    wind.addpoint(ymax,xmax,90.,150.)
    
    wind.addpoint(ymin,xmin,270.,150.)
    wind.addpoint(ymin,xmax,270.,150.)

    return wind

#################################################################

def LatLon2XY(lat, lon):
    x = 60.*lon*abs(np.cos(np.radians(lat)))        # 60 nautical miles per degree longitude at equator, converging toward poles
    y = 60.*lat                                     # 60 nautical miles per degree latitude
    return (x,y)
    
def XY2LatLon(x, y):
    lat = y/60.                                     # 60 nautical miles per degree latitude
    lon = x/(60.*np.cos(abs(np.radians(lat))))      # 60 nautical miles per degree longitude at equator, converging toward poles                             
    return (lat,lon)
def ktsconvert(lst1, lst2):
    for i in range(len(lst1)):
        lst1[i] = lst1[i] * 1.94384                 # 1 m/s = 1.94384 kts
        lst2[i] = lst2[i] * 1.94384
    return lst1,lst2
# the majority of the Pos class code is from AE1205 Python reader
class Pos:
    def __init__(self, xxx_todo_changeme):
        (xcoord, ycoord) = xxx_todo_changeme
        self.x = xcoord
        self.y = ycoord
        return

    def __sub__(self,pos2): # override meaning of minus sign
        rx = self.x - pos2.x
        ry = self.y - pos2.y
        newp = Pos((rx,ry))
        return newp
    def length(self):
        return sqrt(self.x*self.x+self.y*self.y)
    def relLength(self):
        return (self.x*self.x+self.y*self.y)
    def __mul__(self,pos2): # override meaning of asterisk to mean dot product
        return self.x*pos2.x+self.y*pos2.y
    def rounded(self):
        rndx = round(self.x, 1)
        rndy = round(self.y, 1)
        rnd = Pos((rndx,rndy))
        return rnd

#
class Obs:
    def __init__(self,vertices):
        self.vert = vertices
        
        # Ensure obstacle is closed by checking if first and last vertices are equal
        if self.vert[0] != self.vert[-1]:
            self.vert.append(self.vert[0])

        #! need to break apart any non-convex polygons here!        
        
        # if obstacle vertices are not in clockwise order, make them so
        if not self.direction():
            self.vert.reverse()
        return
    def direction(self):
        directioncheck = 0        
        for i in range(len(self.vert)-1):
            directioncheck = directioncheck + (self.vert[i+1][0]-self.vert[i][0])*(self.vert[i+1][1]+self.vert[i][1])
        if directioncheck > 0:
            clockwise = True
        else:
            clockwise = False
        return clockwise
    def intersectroute(self,routedef):
        intersectiontab = []
        (xwpts,ywpts) = list(zip(*routedef.waypoints))

        for i in range(len(routedef.waypoints)-1):                     # iterate through all route segments
            a = Pos((xwpts[i],ywpts[i]))
            b = Pos((xwpts[i+1],ywpts[i+1]))
            for j in range(len(self.vert)-1):                         # iterate through obstacle segments
                c = Pos((self.vert[j][0],self.vert[j][1]))
                d = Pos((self.vert[j+1][0],self.vert[j+1][1]))  
                if notcolinear(a,b,c,d) and intersect(a,b,c,d):                  
                    intersectiontab.append([i,j])                     # i corresponds to route segment and j corresponds to obstacle segment

        return intersectiontab

    def resort(self,segList,parent,intersectTab):
        # vertices of first segment in obstacle segment list:
        ver1 = self.vert[segList[0]]
        ver2 = self.vert[segList[0]+1]

        # vertices of route segment:
        xwpts,ywpts = list(zip(*parent.waypoints))
        rou1 = (xwpts[intersectTab[0][0]],ywpts[intersectTab[0][0]])
        rou2 = (xwpts[intersectTab[0][0]+1],ywpts[intersectTab[0][0]+1])
        # vertices of last segment in obstacle segment list:        
        ver3 = self.vert[segList[-1]]
        ver4 = self.vert[segList[-1]+1]

        # use first vertex of route segment as reference point
        refpt = Pos((rou1))
        # calculate distances between reference point and points of intersection        
        check1 = intersectionpt(ver1,ver2,rou1,rou2) - refpt
        check2 = intersectionpt(ver3,ver4,rou1,rou2) - refpt
        # if distance to intersection point with first obstacle segment is greater
        # than that of last obstacle segment, flip the order of obstacle segment list
        if check1.length() > check2.length():
            segList = list(reversed(segList))
        return segList

    def leftalt(self,segList):                  
        altWptL = []                                               # waypoints along left alternative route
        # (C) populate list of left alternative waypoints (make use of the fact that obstacle vertices are defined in clockwise order)
        for i in range(len(segList)):
            # make exception for last alternative waypoint to prevent "over-rotation" around obstacle
            if i > 0 and  i == len(segList)-1:
                if self.vert[segList[i]] not in altWptL:
                    altWptL.append(self.vert[segList[i]])
            else:
                if self.vert[segList[i]+1] not in altWptL:                
                # append the larger of the two vertices to the list of left alternative waypoints
                    altWptL.append(self.vert[segList[i]+1])
        # (D) add additional obstacle vertices to list of left alternative waypoints, if necessary
        # create empty lists:
        indexLobs = []                                     # indices of obstacle vertices in altWptL
        addL = []                                          # obstacle vertices to add to altWptL
        # check for obstacle vertices in between consecutive altWptL entries
        for i in range(len(altWptL)):
            # populate list of indices of obstacle vertices in altWptL
            indexLobs.append(self.vert.index(altWptL[i]))
            # check if indices are consecutive; if not, need to evaluate further
            if len(indexLobs)>=2 and abs(indexLobs[-1]-indexLobs[-2])%(len(self.vert)-1)>=2:
            # note: total number of obstacle vertices is len(obs)-1 because last vertex is same as first
                # define rngStp as difference between indices
                if indexLobs[-1] < indexLobs[-2]:
                    rngStp = len(self.vert)-indexLobs[-2]+indexLobs[-1]     # need to subtract 1!!!
                else:
                    rngStp = indexLobs[-1]-indexLobs[-2]
                # iterate through obstacle vertices between previous and current altWptL
                for j in range(1,rngStp):
                    # define a and c as previous and current altWptL positions, respectively
                    a = Pos((altWptL[i-1][0],altWptL[i-1][1]))
                    c = Pos((altWptL[i][0],altWptL[i][1]))
                    # define b as "in-between" vertex position, taking into account possible indexing exceptions
                    if self.vert.index(altWptL[i-1])==len(self.vert)-1:
                        bInd = 0
                    else:
                        bInd = self.vert.index(altWptL[i-1])
                    if (bInd+j)>len(self.vert)-1:
                        bArg = (bInd+j)%(len(self.vert)-1)-1
                    else:
                        bArg = bInd+j
                    b = Pos((self.vert[bArg][0],self.vert[bArg][1]))
                    # check direction of a,b,c sequence
                    # append "in-between" vertex b to addL if direction is clockwise (i.e. not counterclockwise)
                    if not ccw(a,b,c):
                        addL.append(self.vert[bArg])        
    ##                    print 'added waypoint'
    ##                else:
    ##                    print 'no point added to altWptL'
                # insert addL list into altWptL
                for ii in range(len(addL)):
                    altWptL.insert(i+ii,addL[ii])
    ##        else:
    ##           print 'altWptR are consecutive vertices of obs'             
        # make sure there are no duplicates in altWptL
        alt = duplicatefilter(altWptL)
        return alt        

    def extupdate(self,alternativepts,destination):
        # exit obstacle as soon as possible to go direct to destination
        alt = list(alternativepts)      
        toremove = []
        # Iterate backwards through alternative waypoints
        for i in range(len(alt)-1):
            a = Pos((alt[-(i+1)-1][0],alt[-(i+1)-1][1]))
            b = destination
          #  b = Pos((destination[0],destination[1]))
            intersection = []
            for j in range(len(self.vert)-1):
                c = Pos((self.vert[j][0],self.vert[j][1]))
                d = Pos((self.vert[j+1][0],self.vert[j+1][1]))
                if intersect(a,b,c,d) and notcolinear(a,b,c,d):
                    intersection.append(j)
                if (a.x, a.y) in self.vert:
                    intersection.append(j)
            if not len(intersection):
#                midptx = float(a.x) + (float(b.x)-float(a.x))/2
#                midpty = float(a.y) + (float(b.y)-float(a.y))/2
#                midpt = Pos((midptx,midpty))
#                if not isInside(midpt,self):
#                    toremove.append(-(i+1))
                toremove.append(-(i+1))
                
            if not len(toremove):
                break
        if len(toremove):
            for i in range(len(toremove)):
                indexfordeletion = toremove[i]                
                del alt[indexfordeletion+i]  
        return alt
        
    def rightalt(self,segList):
        altWptR = []                                               # waypoints along left alternative route
        # (C) populate list of right alternative waypoints (make use of the fact that obstacle vertices are defined in clockwise order)
        for i in range(len(segList)):
                # make exception for last alternative waypoint to prevent "over-rotation" around obstacle
            if i > 0 and  i == len(segList)-1:
                if self.vert[segList[i]+1] not in altWptR:
                    altWptR.append(self.vert[segList[i]+1])
            else:
                if self.vert[segList[i]] not in altWptR:
                # append the smaller of the two vertices to the list of right alternative waypoints                   
                    altWptR.append(self.vert[segList[i]])
        # (D) add additional obstacle vertices to list of right alternative waypoints, if necessary
        # create empty lists:
        indexRobs = []                                         # indices of obstacle vertices in altWptR
        addR = []                                              # obstacle vertices to add to altWptR
        # check for obstacle vertices in between consecutive altWptR entries
        for i in range(len(altWptR)):
            # populate list of indices of obstacle vertices in altWptR
            indexRobs.append(self.vert.index(altWptR[i]))
            # check if indices are consecutive; if not, need to evaluate further
            if len(indexRobs)>=2 and abs(indexRobs[-1]-indexRobs[-2])%(len(self.vert)-1)>=2:
            # note: total number of obstacle vertices is len(obs)-1 because last vertex is same as first
                # define rngStp as difference between indices
                if indexRobs[-1] > indexRobs[-2]:
                    rngStp = len(self.vert)-indexRobs[-1]+indexRobs[-2]       # need to subtract 1!!!
                else:
                    rngStp = abs(indexRobs[-1]-indexRobs[-2])
                # iterate through obstacle vertices between previous and current altWptR
                for j in range(1,rngStp):
                    # define a and c as previous and current altWptR positions, respectively
                    a = Pos((altWptR[i-1][0],altWptR[i-1][1]))
                    c = Pos((altWptR[i][0],altWptR[i][1]))
                    # define b as "in-between" vertex position, taking into account possible indexing exceptions
                    if self.vert.index(altWptR[i-1])==0:
                        bInd = len(self.vert)-1
                    else:
                        bInd = self.vert.index(altWptR[i-1])
                    if (bInd-j)<0:
                        bArg = len(self.vert)-1+(bInd-j)
                    else:
                        bArg = bInd-j
                    b = Pos((self.vert[bArg][0],self.vert[bArg][1]))
                    # check direction of a,b,c sequence
                    # append "in-between" vertex b to addR if direction is counterclockwise
                    if ccw(a,b,c):
                        addR.append(self.vert[bArg])
    ##                    print 'added waypoint'
    ##                else:
    ##                    print 'no point added to altWptR'
                # insert addR list into altWptR
                for ii in range(len(addR)):
                    altWptR.insert(i+ii,addR[ii])
    ##        else:
    ##            print 'altWptR are consecutive vertices of obs'      
        # make sure there are no duplicates in altWptL
        alt = duplicatefilter(altWptR)
        return alt
    
    
    def plotter(self,fig,number, sector_color, plt_vertices_labels=0):
        #create code sequence to draw obstacles (with flexible number of vertices)
        codes = []
        codes.append(Path.MOVETO)

        for j in range(1,len(self.vert)-1):
            codes.append(Path.LINETO)

        codes.append(Path.CLOSEPOLY)

        #draw obstacle
        path = Path(self.vert, codes)
        ax = fig.axes[0]
        patch = patches.PathPatch(path,color=sector_color)
        ax.add_patch(patch)
        #label obstacle (place text somewhere near middle of obstacle)
        obsX = []
        obsY = []

        for i in range(len(self.vert)-1):
            obsX.append(self.vert[i][0])
            obsY.append(self.vert[i][1])
        xytext = (np.mean(obsX)-.1,np.mean(obsY))
        ax.annotate(int(number),xytext)

        # --- Label each vertex with A, B, C, ... ---
        if plt_vertices_labels:
            vertex_labels = self.label_vertices()
            for (x, y), label in vertex_labels.items():
                ax.text(
                    x + 0.05, y + 0.05,   # offset so text doesn’t overlap vertex
                    label,
                    fontsize=10,
                    color="black",
                    weight="bold",
                )

        return

    

    def label_vertices(self):
        """
        Create a dictionary assigning A, B, C, ... to the vertices in self.vert.

        Returns:
            vertex_labels (dict): Mapping like {'A': (x1, y1), 'B': (x2, y2), ...}
        """
        import string
        labels = string.ascii_uppercase  # 'A', 'B', 'C', ...
        vertex_labels = {}

        for i, v in enumerate(self.vert):
            label = labels[i-1] 
            vertex_labels[label] = (v[0], v[1])
        
        # reverse mapping
        vertex_labels = {v: k for k, v in vertex_labels.items()}
        for vertex, label in vertex_labels.items():
            print(f"{label}: vertex {vertex}")
        return vertex_labels

class Route:
#    allroutes = []
    active = []
    def __init__(self,start,end,speed,wyptcoords,distrt,time,deviation):
     #   self.distance = distance
     #   self.heading = heading
        self.start = start
        self.end = end
        self.speed = speed
        self.waypoints = wyptcoords
        self.distance = distrt
        self.speed = speed
        self.deviation = deviation
        self.time = time
#        self.numwpts = len(self.waypoints)
#        Route.allroutes.append(self)
        Route.active.append(self)
        return
    def distancecheck(self):
        runningtotal = []        
        for i in range(len(self.waypoints)-1):
            pt1 = Pos((self.waypoints[i]))            
            pt2 = Pos((self.waypoints[i+1]))
            segment = pt2 - pt1
            runningtotal.append(segment.length())
        return sum(runningtotal)
    def timecheck(self,winddata):
        runningtotal = []        
        for i in range(len(self.waypoints)-1):
            pt1 = Pos((self.waypoints[i]))
            wind1N = winddata[0][i]
            wind1E = winddata[1][i]
            pt2 = Pos((self.waypoints[i+1]))
            wind2N = winddata[0][i+1]
            wind2E = winddata[1][i+1]
            segment = pt2 - pt1
            avgwindN = np.mean([wind1N,wind2N])
            avgwindE = np.mean([wind1E,wind2E])
            heading = atan2(segment.y,segment.x)
            Nspeed = (self.speed * np.sin(heading)) + avgwindN
            Espeed = (self.speed * np.cos(heading)) + avgwindE
            # given heading, how does north/east wind affect speed?
            segspeed = sqrt(Nspeed*Nspeed + Espeed*Espeed)          
            runningtotal.append(segment.length()/segspeed)
        return sum(runningtotal)
    def deviationcheck(self,parentrt,optimizationpriority,winds):
        self.distance = self.distancecheck()
        self.time = self.timecheck(winds)
        if optimizationpriority == 0:
            ownlength = self.distance
            parentlength = parentrt.distance
            return ownlength - parentlength
        elif optimizationpriority == 1:
            owntime = self.time
            parenttime = parentrt.time
            return owntime - parenttime
    def rem(self,wpt2remove):
        #self.waypoints.remove(self.waypoints[wpt2remove]) #had this!
        del self.waypoints[wpt2remove]
#        self.numwpts = len(self.waypoints)
        self.start = self.waypoints[0]
        self.end = self.waypoints[-1]
        return
    def insert(self,index,wpt2add):
        x,y = list(zip(*self.waypoints))
        xroute = list(x)
        yroute = list(y)
        for i in range(len(wpt2add)):
            xroute.insert(index+i,wpt2add[i][0])
            yroute.insert(index+i,wpt2add[i][1])
        self.waypoints = list(zip(xroute,yroute))
#        self.numwpts = len(self.waypoints)
        self.start = self.waypoints[0]
        self.end = self.waypoints[-1]
        return        
    def clean(self,obstacles):
        # REMOVE WAYPOINTS THAT ARE INSIDE GIVEN OBSTACLE(S)
        indexRem = []
        for i in range(len(self.waypoints)):
            for testing in range(len(obstacles)):
                if isInside(Pos((self.waypoints[i])),obstacles[testing]):
                    indexRem.append(i)
        for j in range(len(indexRem)):
    #route.waypoints.remove(route.waypoints[indexRem[i]-i])
    #route.numwpts = route.numwpts - 1
            self.rem(indexRem[j]-j)   
#            self.numwpts = self.numwpts - 1
        return

    def backwardcleanup(self,obstacles,lstptindex):
        # if no obstacle in the way, go direct!
#        toconsider = self.waypoints[0:lstpt+1]        
#        (xwpts,ywpts) = zip(*toconsider)
        loopduration = lstptindex-1             # really it's (lstptindex + 1) - 2 because it's number waypoints - 2            
        flag = 0

        while not flag:
            i = 0
            removed = 0
            while i < loopduration:
                (xwpts,ywpts) = list(zip(*self.waypoints[0:loopduration+2]))
                intersectiontab = []
                a = Pos((xwpts[i],ywpts[i]))
                b = Pos((xwpts[i+2],ywpts[i+2]))
                for index in range(len(obstacles)):            
                    for j in range(len(obstacles[index].vert)-1):                         # iterate through obstacle segments
                        c = Pos((obstacles[index].vert[j][0],obstacles[index].vert[j][1]))
                        d = Pos((obstacles[index].vert[j+1][0],obstacles[index].vert[j+1][1]))  
                        # If the straight line between waypoints i (a) and i+2 (b) intersects 
                        # the obstacle edge (c,d), and the two segments are not just colinear 
                        # lying on top of each other, then the middle waypoint i+1 cannot be 
                        # removed, as it is needed to go around the obstacle. So we add it to the list of intersections.
                        if notcolinear(a,b,c,d) and intersect(a,b,c,d):
                            intersectiontab.append(j) # i corresponds to route segment and j corresponds to obstacle segment

                    if not len(intersectiontab):
#                        rndobs = []
#                        for i in range(len(obstacles[index].vert)):
#                            rnded = Pos(obstacles[index].vert[i]).rounded()
#                            rndobs.append(rnded)
#                            print rndobs
#                        if (a.rounded() in rndobs) and (b.rounded() in rndobs):   
#                            print 'got in!'
                        midptx = float(a.x) + (float(b.x)-float(a.x))/2
                        midpty = float(a.y) + (float(b.y)-float(a.y))/2
                        midpt = Pos((midptx,midpty))
                        skip_check_inside = 0
                        for l in range(len(obstacles[index].vert)-1):
                            vectorA = midpt-Pos(obstacles[index].vert[l])
                            vectorB = midpt-Pos(obstacles[index].vert[l+1])
                            # if either vector has length 0, the midpoint between two waypoints 
                            # in the trajectory is on a vertex of the obstacle. Therefore the 
                            # waypoint in the middle can be removed and we can skip the isInside check
                            if vectorA.length() == 0 or vectorB.length() == 0:
                                skip_check_inside = 1
                        if not skip_check_inside:
                            if isInside(midpt,obstacles[index]):
                                intersectiontab.append(index)
                if not len(intersectiontab):
                    self.rem(i+1)
                    removed = 1
                    loopduration = loopduration-1
                i = i+1
            if not removed:
                flag = 1
        return

def parse(start,end):            # start and end inputs are from Pos class
    # calculate straight-line distance between (x0,y0) and (xdest,ydest)
    trigger = 0    
    delta = end - start
    totalDist = delta.length()
    waypointinterval = float('inf')                            # define average NM between nominal waypoints
#    waypointinterval = 30.    
    # define number of waypoints
    if waypointinterval < totalDist and waypointinterval != 0:
        numWpts = int(totalDist/waypointinterval)-1
        # calculate distance between waypoints and heading along route
        distWpt = totalDist/(numWpts+1)
        trigger = 1
    heading = atan2(delta.y,delta.x)                            # use atan2 function for quadrant universality
    # initialize x and y waypoint lists with origin as first waypoint    
    xwpts = [start.x]
    ywpts = [start.y]
    if trigger:
        # add intermdediate waypoints to the list     
        for i in range(1,numWpts+1):
            xnew = xwpts[-1] + (distWpt*np.cos(heading))
            xwpts.append(xnew)
            ynew = ywpts[-1] + (distWpt*np.sin(heading))
            ywpts.append(ynew)
    # add the destination as last waypoint    
    xwpts.append(end.x)                                         #! waypoints should eventually have associated speeds and altitudes...
    ywpts.append(end.y)
    # store route information in dictionary structure
    return totalDist, heading, list(zip(xwpts,ywpts))
    
def callWinds(wind,route,altitude=0):
    # Call wind, with list for lat,lon and altitude:
    X,Y = list(zip(*route.waypoints))
    X = list(X)
    Y = list(Y)
    Lats = []
    Lons = []
    for i in range(len(X)):
        lat,lon = XY2LatLon(X[i],Y[i])
        Lats.append(lat)
        Lons.append(lon) 
    # Using vector for position but all on same altitude
    if altitude == 0:
        vn,ve = wind.getdata(Lats,Lons)
    else:
        vn,ve = wind.getdata(Lats,Lons,altitude)
    windsaloft = (vn,ve)                                         # windsaloft is vn, ve in kts   
    return windsaloft
    
def detSign(vector1,vector2):
    det = float(vector1.x)*float(vector2.y)-float(vector1.y)*float(vector2.x)
    if det != 0:
        dsign = int(det/abs(det))
    else:
        dsign = 1
    return dsign
    
def isInside(position,obstacle):
    angle = []
    for i in range(len(obstacle.vert)-1):
        vectorA = position-Pos(obstacle.vert[i])
        vectorB = position-Pos(obstacle.vert[i+1])

        # Equation (2) of Hormann 2001 Computational Geometry publication
        # computationally expensive version of calculating winding number
        cos_angle = round((vectorA * vectorB) / (vectorA.length() * vectorB.length()),12)  # round to avoid floating point precision issues

        angle.append(acos(cos_angle) * detSign(vectorA, vectorB))

    if int(round(sum(angle)/(2*pi))) == 0:
        inside = False
    else:
        inside = True
    
    return inside

# use counterclockwise (ccw) and intersect definitions from Bryce Boe's line segment intersection check
# found online (http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/)
# ccw and intersect definitions take in Pos() output for A, B, and C
#def ccw(A,B,C):
#	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
#def intersect(A,B,C,D):
#	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
 
def ccw(A,B,C):
    return (C.y-A.y)*(B.x-A.x) >= (B.y-A.y)*(C.x-A.x)
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
def notcolinear(A,B,C,D):
    if (C.y-B.y)*(A.x-B.x) - (A.y-B.y)*(C.x-B.x) == 0:
        return False
    elif (D.y-A.y)*(B.x-A.x) - (B.y-A.y)*(D.x-A.x) == 0:
        return False
    else:
        return True
    
  #  return bool(round((C.y-A.y)*(B.x-A.x) - (B.y-A.y)*(C.x-A.x))) and bool(round((D.y-A.y)*(B.x-A.x) - (B.y-A.y)*(D.x-A.x)))

# use point-slope form of line equation to find intersection point
# (only for use on line segments that we already know will intersect)
def intersectionpt(a,b,c,d):
    slope1 = (a[1]-b[1])/(a[0]-b[0])
    slope2 = (c[1]-d[1])/(c[0]-d[0])
    x = (slope1*a[0] - slope2*c[0] + c[1] - a[1]) / (slope1 - slope2)
    y = slope1*(x-a[0]) + a[1]
    intpt = Pos((x,y))
    return intpt
    
def wyptannotate(xwpts,ywpts):
    for i in range(len(xwpts)):
        plt.scatter(xwpts[i],ywpts[i])
        wptname = 'wpt' + str(i)
        if i==0:
            plt.annotate('(x0,y0)',(xwpts[i],ywpts[i]))
        elif i==(len(xwpts)-1):
            plt.annotate('(xdest,ydest)',(xwpts[i],ywpts[i]))
        else:
            plt.annotate(wptname,(xwpts[i],ywpts[i]))
            
def duplicatefilter(lst):
   unique = []
   for i in lst:
       if i not in unique:
           unique.append(i)
   return unique   