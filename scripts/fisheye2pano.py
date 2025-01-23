#!/usr/bin/python3
#
# NOTE: code adapted from https://github.com/kscottz/dewarp.git, which was
# designed for the RPi
import argparse as argp
import glob
import json
import math
#assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import os
import sys
import time

import cv2
import numpy as np
import perception as yolo
import pandas as pd

UNDEFINED_VALUE = float(-9999999.9)


# deprecated, checks if point in the sphere is in our output
def isInROI(x,y,R1,R2,Cx,Cy):
    isInOuter = False
    isInInner = False
    xv = x-Cx
    yv = y-Cy
    rt = (xv*xv)+(yv*yv)
    if( rt < R2*R2 ):
        isInOuter = True
        if( rt < R1*R1 ):
            isInInner = True
    return isInOuter and not isInInner

# build the mapping from DESTINATION back to SOURCE (360 fisheye image)
def buildMap(Wd,Hd,R_1,R_2,Cx,Cy):
    map_x = np.zeros((Hd,Wd),np.float32)
    map_y = np.zeros((Hd,Wd),np.float32)
    for y in range(0,int(Hd-1)):
        print("dest row: {0}".format(y),end="\r")
        Xp = np.linspace(0.0,Wd,num=Wd)
        r = (float(y)/float(Hd))*(R_2-R_1)+R_1
        theta = (Xp/float(Wd))*2.0*np.pi
        xS = Cx+r*np.sin(theta)
        yS = Cy+r*np.cos(theta)
        map_x[y,:Wd] = xS[:Wd]
        map_y[y,:Wd] = yS[:Wd]
    return map_x, map_y



# do the unwarping 
def unwarp(img,xmap,ymap):
    result = cv2.remap(img,xmap,ymap,cv2.INTER_LINEAR)
    return result

def main(): 
    xmap = []
    ymap = []
    Cx = UNDEFINED_VALUE
    Cy = UNDEFINED_VALUE
    Rad1 = UNDEFINED_VALUE
    Rad2 = UNDEFINED_VALUE
    Hd = UNDEFINED_VALUE
    Wd = UNDEFINED_VALUE

    first_time = bool(True)

    parser = argp.ArgumentParser(prog=sys.argv[0], description="Rectify fisheye images using estimated intrinsic parameters from the camera calibration.")

    parser.add_argument('-v', '--videosource', type=str, required=True, default=[],
                        help='FULL path of avi file')
    parser.add_argument('-i', '--inputimg', type=str, required=False, default=[],
                        help='FULL path to a single input image')
    parser.add_argument('-s', '--start_nv', type=int, required=True, default=[],
                        help='Starting number of views')
    parser.add_argument('-e', '--end_nv', type=int, required=True, default=[],
                        help='Ending number of views')
    args=parser.parse_args()

    # cv2.namedWindow("fisheye input", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("Panoramic", cv2.WINDOW_AUTOSIZE)

    # get height and width of the panoramic named window
    cap=cv2.VideoCapture(args.videosource)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    detection_and_tracking = yolo.DetectandTrackingModule(fps=fps)
    # calib = json.load(open('fisheye_calibration_data.json'))
    dt = 0.0
    while(cap.isOpened()):
        result,img=cap.read()
        if result==False:
            break

        if first_time == True:
            first_time = False
            Ws = img.shape[0]
            Hs = img.shape[1]
            if Cx == UNDEFINED_VALUE:
                Cx = Ws/2
            if Cy == UNDEFINED_VALUE:
                Cy = Hs/2
            if Rad1 == UNDEFINED_VALUE:
                Rad1 = int(Ws*0.20)
            if Rad2 == UNDEFINED_VALUE:
                Rad2 = int(Ws*0.45)
            if Wd == UNDEFINED_VALUE:
                Wd = int(2.0*((Rad2+Rad1)/2)*np.pi)
                #Wd = int(0.95*Wd)
            if Hd == UNDEFINED_VALUE:
                Hd = int(Rad2-Rad1)
            print("BUILDING MAP to src ({0}x{1}) for {2}x{3} panoramic image...".format(Ws,Hs, Wd,Hd))
            xmap,ymap = buildMap(Wd,Hd,Rad1,Rad2,Cx,Cy)
            print("MAP DONE!")
            # Calculate the resized dimensions
            resized_width = int(Wd * 0.25)
            resized_height = int(Hd * 0.25)

            # Initialize video writer with the resized dimensions
            panoramic_video = cv2.VideoWriter('panoramic_calib.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (resized_width, resized_height))

        pano_img = unwarp(img, xmap,ymap)
        
        # flip about x-axis
        pano_img2 = cv2.flip(pano_img, 0)
        pano_sc = cv2.resize( pano_img2, (0,0), fx=0.25, fy=0.25 )
        img_sc = cv2.resize( img, (0,0), fx=0.25, fy=0.25 )
        # cv2.imshow("fisheye input", img_sc)
        # cv2.imshow("Panoramic",  pano_sc)
        panoramic_video.write(pano_sc)

        # create N 'wide-angle camera' views 
        Nviews = 4
        HOV = np.deg2rad(100)
        Hres_pano = pano_img.shape[1]
        Hres = int(Hres_pano*HOV/(np.pi*2))
        print("Hres = ", Hres)
        ROIimages = []
        Vres = pano_img.shape[0]
        # trackers = tracker.ObjectTracker(max_disappeared=30,max_trail_length=30)

        # Frame details (update these based on your actual frame size and desired FPS)
        # frame_width = 640  # Use the width of the annotated frame
        # frame_height = 640  # Use the height of the annotated frame
        # fps = 10  # Adjust FPS as needed
        # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))
        for nv in range(args.start_nv, args.end_nv):
            xspan = nv*Hres
#            print("#",nv, "=",xspan, xspan+Hres)
            if nv > 0:
                xspan += 1

            if xspan+Hres < Hres_pano:
                roi = pano_img2[0:Vres-1, xspan:xspan+Hres, :]
            else: # wrap start of pano
                overlap = xspan+Hres- Hres_pano
                roi = cv2.hconcat([pano_img2[0:Vres-1, xspan:Hres_pano, :], pano_img2[0:Vres-1, 0:overlap, :] ])

            roi_sc = cv2.resize(roi, (0,0), fx=0.25, fy=0.25 )

            if frame_count % 300 == 0:
                    dt += 1
                    annotated_frame,df_existing = detection_and_tracking.detect_and_track(roi,nv,dt)
                    # Display the annotated frame
                    # cv2.imshow("YOLO11 Tracking", annotated_frame)
                    # cv2.waitKey(0)
            else:
                print("No detections found in the frame.")
                annotated_frame = detection_and_tracking.resize_frame(roi, target_size=640)

            # Check if there are any detections
            cv2.imshow("Annotated {0}-{1}".format(xspan, xspan + Hres), annotated_frame)
            cv2.waitKey(1)
            # out.write(annotated_frame)
            # cv2.imshow("{0}-{1}".format(xspan,xspan+Hres), roi_sc)

        k = cv2.waitKey(2)
        if k & 0xFF ==ord('q'):
            # df_existing.to_csv('tracking_data.csv', index=False)
            break
        
    cap.release()
    # panoramic_video.release()
    # out.release()
    cv2.destroyAllWindows()

    # df_existing.to_csv('tracking_data.csv', index=False)
    # print(f"Output video saved as {output_video_path}")
    sys.exit(0)
    
if __name__ == '__main__':
    main()

# Show the user a frame let them left click the center
# of the "donut" and the right inner and outer edge
# in that order. Press esc to exit the display


# 0 = xc yc
# 1 = r1
# 2 = r2
# center of the "donut"    
#Cx = vals[0][0]
#Cy = vals[0][1]
# Inner donut radius
#R1x = vals[1][0]
#R1y = vals[1][1]
#R1 = R1x-Cx
# outer donut radius
#R2x = vals[2][0]
#R2y = vals[2][1]
#R2 = R2x-Cx
# our input and output image siZes
#Wd = 2.0*((R2+R1)/2)*np.pi
#Hd = (R2-R1)
#Ws = img.width
#Hs = img.height

# build the pixel map, this could be sped up

# do an unwarping and show it to us
#result = unwarp(img,xmap,ymap)


# SimpleCV/OpenCV video out was giving problems
# decided to output frames and convert using
# avconv / ffmpeg. Comment out the block below
# to save to video

#ofname = 'OUT.AVI'
#vs = VideoStream(fps=20,filename=ofname,framefill=False)
#vs.initializeWriter((640,480))

# I used these params for converting the raw frames to video
# avconv -f image2 -r 30 -v:b 1024K -i samples/lapinsnipermin/%03d.jpeg output.mpeg
#i = 0
#while img is not None:
#    print img.width,img.height
#    result = unwarp(img,xmap,ymap)
#    #derp = result.adaptiveScale(resolution=(640,480))
#    #result = result.resize(w=img.width)
#    # Once we get an image overlay it on the source
#    derp = img.blit(result,(0,img.height-result.height))
#    derp = derp.applyLayers()
#    #derp = derp.resize(640,480)
#    derp.save(disp)
#    # Save to file
#    fname = "FRAME{num:05d}.png".format(num=i)
##    derp.save(fname)
#    #vs.writeFrame(derp)
#    # get the next frame
#    img = vc.getImage()
#    i = i + 1

