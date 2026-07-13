"""
Adapted from D. Rein-Weston

2D example of aircraft at (x0,y0) with destination of (xdest,ydest),
and assumed constant speed and altitude (parameters defined in "odfile.txt").
Straight-line route is created with waypoints spaced by an average number
of nautical miles (defined in the parse function in "mytools.py").
Obstacles of varying number of vertices are read-in from obstacle definition
(eg, SC1_multi_convex.txt) file. Line segments that define obstacles are
checked to see if there are any intersections with route segments.
The branching algorithm proceeds by branching on first encountered obstacle,
creating clockwise and counterclockwise route alternatives, and selecting
the most "promising" route (i.e. the route requiring least deviation from its
parent route) first for further branching. Less-promising routes are later
returned to and evaluated. The most efficient route option is plotted.

Input settings include windsOn parameter and optimizationpriority parameter.
By setting the optimization priority to 0 or 1, the algorithm evaluates a
distance-efficient or time-efficient route, respectively.

Reads from origin/destination and obstacle definition text files
Uses tools from "mytools.py" and "windfield.py".

"""

import matplotlib.pyplot as plt
import numpy as np
from bluesky_gym.envs.common.tools_deterministic_path_planning import Pos, LatLon2XY, XY2LatLon, Obs, parse,\
Route,wyptannotate,callWinds,intersectionpt,specifywindfield

def det_path_planning(lat0, lon0, altitude, TAS, latdest, londest, inputObs):
    #################################################################
    # PRINTING OPTIONS

    #################################################################
    # enable printing for debugging purposes
    debugging_printing_flag = 0
    # enable logging on console
    console_logging_flag   = 0
    # define whether or not to use wind
    windsOn                 = 0
    # Define optimization strategy
    optimizationpriority    = 0       # type 0 for distance, 1 for time
    #################################################################

    #PLOTTING OPTIONS

    #################################################################
    # enable plotting
    plt_enable            = 0
    plt_vertices_labels = 0

    if debugging_printing_flag:
        console_logging_flag   = 1
        plt_enable = 1
        plt_vertices_labels = 1

    if not plt_enable:
        # plot direct route
        pltdirectrt             = 0
        # plot all trial solutions
        plttrialsols            = 0
        # plot final incumbent
        pltfinalsol             = 0
    else:
        # plot direct route
        pltdirectrt             = 1
        # plot all trial solutions
        plttrialsols            = 1
        # plot final incumbent
        pltfinalsol             = 1
        # show console logging
        console_logging_flag    = 1

    #################################################################
    # DEFINE ORIGIN, DESTINATION, SPEED

    #process the data:
    # (lat0,lon0) =           lst[0]                          # initial latitude and longitude
    # (latdest,londest) =     lst[1]                          # destination latitude and longitude

    # # process third input as altitude, note: altitude taken as constant throughout trajectory
    # altitude =              lst[2][0]/3.28084               # altitude converted to [m] for windfield.py
    # # process fourth input as true airspeed (TAS), note: TAS taken as constant throughout trajectory
    # TAS =                   lst[2][1]

    # convert origin and destination to x,y coordinates
    origin =                Pos(LatLon2XY(lat0,lon0))
    destination =           Pos(LatLon2XY(latdest,londest))

    # DEFINE OBSTACLE(S)

    # While processing obstacle vertices in for loops, also define
    # boundaries of windfield
    orig,dest = [lat0, lon0],[latdest,londest]
    ymin = min(orig[0],dest[0])
    ymax = max(orig[0],dest[0])
    xmin = min(orig[1],dest[1])
    xmax = max(orig[1],dest[1])
    
    obstacle_list_xy = []
    for polygon in inputObs:
        vertices_list_xy = []
        for lat, lon in polygon:
            x, y = LatLon2XY(lat, lon)
            vertices_list_xy.append((x, y))
        obstacle_list_xy.append(vertices_list_xy)

    # create obstacle dictionary (key is obstacle index)
    obsDic_xy = {i: obstacle_list_xy[i] for i in range(len(obstacle_list_xy))}
    
    # define as instance of the obstacle class
    for i in range(len(obsDic_xy)):
        obsDic_xy[i] = Obs(obsDic_xy[i])

    # Add edge around it of 1%
    margin = 0.01
    xspan = xmax-xmin
    yspan = ymax-ymin

    xmin = xmin - margin*xspan
    xmax = xmax + margin*xspan
    ymin = ymin - margin*yspan
    ymax = ymax + margin*yspan
    
    wind = specifywindfield(ymin,ymax,xmin,xmax)

    #################################################################

    # GENERATE DIRECT ROUTE and PRINT ROUTE INFO

    #################################################################

    # generate direct route using origin and destination info
    distance, heading, wypts = parse(origin,destination)
    nominaltime = distance/TAS
    directrt = Route(origin,destination,TAS,wypts,distance,nominaltime,0.0)

    # print distance and heading info
    if console_logging_flag:
        print('direct route info:')
        print('distance       ', distance, ' [NM]')
        print('true airspeed  ', TAS, '         [kts]')
        print('time (no wind) ', nominaltime, '[hr]')
        #print 'heading is', degrees(heading), 'degrees'
        print('')
        if optimizationpriority == 0:
            print('optimizing for DISTANCE')
        elif optimizationpriority == 1:
            print('optimizing for TIME')
        print('')
        print('############## branching algorithm ##############')
        print('')

    # REMOVE DIRECT ROUTE WAYPOINTS THAT ARE INSIDE ANY OBSTACLE

    directrt.clean(obsDic_xy)

    Route.active[-1] = directrt
    # Route.active.append(directrt)

    #################################################################

    # MAKE PLOT SHOWING DIRECT ROUTE AND OBSTACLES

    #################################################################

    # create plot
    if plt_enable:
        fig = plt.figure()

        # define waypoints along direct route, annotate and plot
        (xwpts,ywpts) = list(zip(*directrt.waypoints))               # unzip x and y waypoint coordinates
        wyptannotate(xwpts,ywpts)
        #plt.plot(xwpts,ywpts,'--')

    if plt_enable:
        # label figure
        fig.suptitle('2D Avoidance', fontsize=20)
        ax = fig.axes[0]
        plt.xlabel('X: position w.r.t. Prime Meridian [NM]')
        # plt.xlim([])
        plt.ylabel('Y: position w.r.t. Equator [NM]')

        # for all obstacles: draw and label (with a number according to order defined)
        sector_color = ['b', 'c', 'g', 'm', 'r', 'sienna','y', 'orange', 'pink', 'purple', 'lime', 'dodgerblue', 'olive', 'gold', 'coral', 'teal', 'orchid', 'slateblue', 'tomato', 'violet']
        color_counter = 0
        for cur in range(len(obsDic_xy)):
            obsDic_xy[cur].plotter(fig,cur, sector_color[color_counter], plt_vertices_labels)
            color_counter += 1

    
    #################################################################

    # IF INTERSECTION OF ROUTE SEGMENTS WITH OBSTACLE SEGMENTS EXISTS,
    # CREATE ALTERNATIVE ROUTES

    #################################################################

    incflag = 0
    incumbent = float('inf')

    while len(Route.active): 
        if console_logging_flag:
            print('length of active: ',len(Route.active))

        if len(Route.active) > 1000:
            if plt_enable:
                plt.show()
            Route.active = []
            raise Exception("Impossible to find a route")
        
        directrtplted = 0
        
        ## logic for setting parent to least deviation OR only option from active list
        if len(Route.active) == 1:
            parent = Route.active.pop()
            # if there is only 1 active route and no incumbent,
            # assume it's direct rt and plot for reference        
            if incflag == 0 and pltdirectrt:
                parentX,parentY = list(zip(*parent.waypoints))
                plt.plot(parentX,parentY,'--') 
                directrtplted = 1
        else:
            deviationmin = []
            for i in range(len(Route.active)):
                deviationmin.append(Route.active[i].deviation)
            topop = deviationmin.index(min(deviationmin))
            parent = Route.active.pop(topop)
        ##
        if plttrialsols and not directrtplted:    
            parentX,parentY = list(zip(*parent.waypoints))
            plt.plot(parentX,parentY,'--')

            # if debugging_printing_flag:
            #     print('parentX', parentX)
            #     print('parentY', parentY)


        # if there is an incumbent... check if route length (or time) is greater than incumbent length (or time)
        if incflag==1:
            if optimizationpriority == 0:
                if parent.distance>=incumbent.distance:
                    # "fathom route" because it can only get longer with deviations
                    if console_logging_flag:
                        print('> incumbent')
        #                print 'parent distance [NM]:', parent.distance
                        print('')
                    continue
            elif optimizationpriority == 1:
                if parent.time>=incumbent.time:
                    # "fathom route" because it can only get longer with deviations
                    if console_logging_flag:
                        print('> incumbent')
        #                print 'parent time [hr]:', parent.time
                        print('')
    #                parentX,parentY = zip(*parent.waypoints)
    #                plt.plot(parentX,parentY,'-.')   
                    continue

    #    Parentx,Parenty = zip(*parent.waypoints)
    #    plt.plot(Parentx,Parenty,'--')     


        # check if parent route intersects any obstacles
        allintersections =  []                          # create empty list to hold lists for each obstacle       
        for obstacle in range(len(obsDic_xy)):             # loop through all obstacles
            # tabulate intersections between route segments and obstacle segments
            allintersections.append(obsDic_xy[obstacle].intersectroute(parent))

        if debugging_printing_flag:
            print(f'allintersections with obstacle ', allintersections)
            for obstacle in range(len(allintersections)):
                if allintersections[obstacle]:
                    print(f'all intersections with obstacle {obstacle} (route segment, obstacle segment): ', allintersections[obstacle])
        # if there are intersections (if len is the same as if len > 0), define branching obstacle as first encountered
        if len([_f for _f in allintersections if _f]):
            # find index of obstacle in obsDic that will be encountered first
            firstseg = []
            for obst in range(len(allintersections)):
                rseg = []
                if allintersections[obst]:  
                    for i in range(len(allintersections[obst])):
                        # append the route segment index to rseg list
                        rseg.append(allintersections[obst][i][0])
                        if debugging_printing_flag:
                            print(f'rseg', rseg)
                    # append the minimum  to firstseg list
                    firstseg.append(min(rseg))
                else:
                    # firstseg.append('NaN')
                    firstseg.append(np.nan)
            # find index of minimum route segment in firstseg list
            if debugging_printing_flag:
                print(f'firstseg', firstseg)

            # if the minimum value is in the list multiple times, calculate
            # intersection pts of each obstacle within that route segment and
            # select obstacle with least distance between route segment start and
            # point of intersection with obstacle
            valid_firstseg = [f for f in firstseg if isinstance(f, int)]
            if firstseg.count(min(valid_firstseg)) > 1:
                indices = [i for i, x in enumerate(firstseg) if x == min(valid_firstseg)]
                distlist = []
                options = []
                refpt = Pos(parent.waypoints[min(valid_firstseg)])
                routpt1 = parent.waypoints[min(valid_firstseg)]
                routpt2 = parent.waypoints[min(valid_firstseg)+1]
                for i in range(len(indices)):
                    for j in range(len(allintersections[indices[i]])):
                        a = obsDic_xy[indices[i]].vert[allintersections[indices[i]][j][1]]
                        b = obsDic_xy[indices[i]].vert[allintersections[indices[i]][j][1]+1]
                        checkdist = intersectionpt(a,b,routpt1,routpt2) - refpt                  
                        distlist.append((checkdist.length()))               
                    options.append(min(distlist))
                branch = indices[options.index(min(options))]

            # else just take index of minimum route segment in firstseglist
            else:
                branch = firstseg.index(min(valid_firstseg))

            # print which obstacle has been selected as branching obstacle
            if console_logging_flag:
                print('branch obstacle', branch)
                print('')
            
            intersectTab = allintersections[branch]
            if debugging_printing_flag:
                print(f'intersectTab', intersectTab)

            # (A) populate list of obstacle segments that intersect with route segment
            segList = []                                           # empty list
            for index in range(len(intersectTab)):
                segList.append(intersectTab[index][1])             # segList contains the smaller of the two obstacle indices defining the segment
            if debugging_printing_flag:
                print(f'segList', segList)
            
            # (B) re-sort obstacle segment list here, in order of first-encountered       
            segList = obsDic_xy[branch].resort(segList,parent,intersectTab)
            if debugging_printing_flag:
                print(f'segList resorted', segList)
            
            # (C) populate lists of left and right alternative waypoints
            # (make use of the fact that obstacle vertices are defined in clockwise order)

            if debugging_printing_flag:
                vertices_labels = obsDic_xy[branch].label_vertices()

            altWptL = obsDic_xy[branch].leftalt(segList)
            altWptR = obsDic_xy[branch].rightalt(segList)

            if debugging_printing_flag:
                print("---")
                for pt in altWptL:
                    label = vertices_labels.get((pt[0], pt[1]), None)
                    if label:
                        print(f"altWptL contains {label}: {pt}")
                print('---')
                for pt in altWptR:
                    label = vertices_labels.get((pt[0], pt[1]), None)
                    if label:
                        print(f"altWptR contains {label}: {pt}")
                print("---")

            # clean up alternative waypoint lists to ensure no repeated waypoints
            altWptLclean = obsDic_xy[branch].extupdate(altWptL,destination)
            altWptRclean = obsDic_xy[branch].extupdate(altWptR,destination)            

            if debugging_printing_flag:
                print("---")
                for pt in altWptL:
                    label = vertices_labels.get((pt[0], pt[1]), None)
                    if label:
                        print(f"altWptLclean contains {label}: {pt}")
                print("---")
                for pt in altWptR:
                    label = vertices_labels.get((pt[0], pt[1]), None)
                    if label:
                        print(f"altWptRclean contains {label}: {pt}")
                print("---")

            # (D) CREATE TRIAL ROUTES WITH PREVIOUSLY CALCULATED ALTERNATIVE WAYPOINTS
            # check which route segment contained the first intersection, insert alternative waypoints       
            altWptIndex = intersectTab[0][0] + 1                    # note: this works because no obstacle will span more than one route segment

            # ROUTE L
            routeL = Route(origin,destination,TAS,parent.waypoints,parent.distance,parent.time,parent.deviation)
            routeL.insert(altWptIndex,altWptLclean)

            # backward cleanup
            lstpt = altWptIndex+len(altWptLclean)                  # define the last waypoint index to consider (exit wypt from branching obstacle)
            routeL.backwardcleanup(obsDic_xy,lstpt)

            if windsOn == 1:
                windsaloftL = callWinds(wind,routeL) 
            else:
                windsaloftL = [np.zeros(len(routeL.waypoints)),np.zeros(len(routeL.waypoints))]

            routeL.deviation = routeL.deviationcheck(parent,optimizationpriority,windsaloftL)

            # should consider somehow adding the alt waypoints (and doing any other necessary changes, i.e. backward cleanup)
            # and THEN declaring it's a route (and thus adding it to active waypoint list)
            # need to do this as well for the routeR and directrt
            # Route.active[-1] = routeL
            Route.active.append(routeL)

            # plot left route in red        
    #        leftpltX,leftpltY = zip(*routeL.waypoints)
    #        plt.plot(leftpltX,leftpltY,':',color='r')

            # ROUTE R        
            routeR = Route(origin,destination,TAS,parent.waypoints,parent.distance,parent.time,parent.deviation)
            routeR.insert(altWptIndex,altWptRclean)

            # backward cleanup
            lstpt = altWptIndex+len(altWptRclean)                  # define the last waypoint index to consider (exit wypt from branching obstacle)        
            routeR.backwardcleanup(obsDic_xy,lstpt)
            
            if windsOn == 1:
                windsaloftR = callWinds(wind,routeR)    
            else:
                windsaloftR = [np.zeros(len(routeR.waypoints)),np.zeros(len(routeR.waypoints))]
            
            routeR.deviation = routeR.deviationcheck(parent,optimizationpriority,windsaloftR)
            # Route.active[-1] = routeR
            Route.active.append(routeR)
            
            # plot right route in green        
    #        rightpltX,rightpltY = zip(*routeR.waypoints)
    #        plt.plot(rightpltX,rightpltY,'--',color='g')  


        # (E) plot alternative waypoints (right waypoints in green, left waypoints in red)
            

    #        for i in range(len(altWptR)):
    #            plt.scatter(altWptR[i][0],altWptR[i][1],color='green')
    #        for i in range(len(altWptL)):
    #            plt.scatter(altWptL[i][0],altWptL[i][1],color='red')
                
        else:
            if console_logging_flag:
                print("no route intersections!")
            
            # update incumbent if either no incumbent exists
            # or if this route (with no intersections) is shorter than current incumbent
            if incflag == 0:
                incumbent = parent
                if console_logging_flag:
                    print('* first incumbent')
                    print('')
                incflag = 1
            elif optimizationpriority == 0:
                if parent.distance < incumbent.distance:
                    incumbent = parent
                    if console_logging_flag:
                        print('--> incumbent updated')
                        print('')
            elif optimizationpriority == 1:
                if parent.time < incumbent.time:
                    incumbent = parent
                    if console_logging_flag:
                        print('--> incumbent updated')
                        print('')

    if console_logging_flag:
        print('############## end algorithm ##############')
        print('')

    IncX,IncY = list(zip(*incumbent.waypoints))
    

    if pltfinalsol:
        plt.plot(IncX,IncY,'-')  

    if console_logging_flag:
        if optimizationpriority == 0:
            print('optimized route distance [NM]: ', incumbent.distance)
            print('')
        elif optimizationpriority == 1:
            print('optimized route time [hr]: ', incumbent.time)
            print('')
            
    waypoint_latlon = []
    for element in incumbent.waypoints:
        waypoint_latlon.append(XY2LatLon(element[0], element[1]))

    # import code
    # code.interact(local= locals())
    # SHOW INTERACTIVE PLOT WINDOW
    if plt_enable:
        plt.show()
    
    return waypoint_latlon