#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv2
import os
# import video
from common import draw_str

lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                 )

feature_params = dict(maxCorners=50,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture("data/1.mp4")
        # self.cam = cv.VideoCapture("CopterPixPro.mp4")
        self.folder = "./data/t2/"
        self.frame_idx = 0

    def load_images_from_folder(self):
        images = []
        for filename in os.listdir(self.folder):
            img = cv2.imread(os.path.join(self.folder, filename))
            if img is not None:
                images.append(img)
        return images

    def run(self):
        images = os.listdir(self.folder)
        count=0
        total_x = 0
        total_y = 0
        while True:
            _ret, frame = self.cam.read()
            # frame = cv.imread(os.path.join(self.folder, images[count]))
            count += 1
            frame = cv2.resize(frame, (1280, 720))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            vis_dip = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # print(np.nansum(_err)/len(_err))
                # print(_err)

                # print(p1.shape)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(axis=-1)
                good = d < 1

                new_tracks = []
                old_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    good_flag = True
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                m = None
                if len(self.tracks)>0 and len(p0)>0:

                    idx = np.where(_st == 1)[0]
                    prev_pts = p0[idx]
                    curr_pts = p1[idx]

                    assert prev_pts.shape == curr_pts.shape

                    m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

                # The Translation
                if m is not None:
                    tx = m[0, 2]
                    ty = m[1, 2]
                    total_x += tx
                    total_y += ty
                    # print("THE TRANSLATION X:",tx)
                    # print("THE TRANSLATION Y:",ty)
                    # print("THE total X:", total_x, "THE total Y:",total_y)
                    total_x_direction = "left" if total_x>0 else "right"
                    curr_x_direction =  "left" if tx>0 else "right"
                    draw_str(vis_dip, (20, 30), 'Total X translation : ' + str(round(total_x,1)) +" " +
                             total_x_direction +'    Total Y translation : ' +  str(round(total_y,1)),font_scale=2.0)
                    draw_str(vis_dip, (70, 60), 'Current X translation : ' + str(round(tx,2)) + " to the "+ curr_x_direction+
                             '                          Current Y translation : ' +  str(round(ty,2)),font_scale=1.2,thickness=1)



                    # Extract rotation angle from the rotation matrix. arctan(sin/cos) = a
                    angle = np.arctan2(m[1, 0], m[0, 0])
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                # for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                #     cv.circle(mask, (x, y), 5, 0, -1) # 0 = black color. -1 thickness full. 5 is the radius
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            final_img = vis_dip.copy()
            #final_img = np.hstack((vis_dip,vis))
            #final_img = cv2.resize(final_img, (1680, 844))

            # cv.imshow('lk_track', vis)
            cv2.imshow('dip', final_img)


            ch = cv2.waitKey(1)
            if ch == 27:
                break


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
