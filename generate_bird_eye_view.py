import cv2
import numpy as np
import pandas as pd
import pickle

from copy import deepcopy

from birdeyeview.bird_eye_view import *

# animation output dimension
output_width = 1000
pad = 0.22
output_height = int(output_width * (1 - pad) * 2 * (1 + pad))

# coordinates of the 4 corners of the court
image_pts = np.array([(561, 259), 
	                  (1261, 259),
	                  (1618, 913),
	                  (216, 913)]).reshape(4, 2)
bev_pts = np.array(court_coor(output_width, pad)).reshape(4, 2)

# homography matrix to go from real world image to bird eye view
M = transition_matrix(image_pts, bev_pts)

# players positions in bird eye view
test = {'cp_0': [], 'cp_1': []}

# players positions in bird eye view
positions_df = pd.read_csv('tracking_players.csv')
player_1_pos = positions_df.loc[positions_df["id"]==1]
player_2_pos = positions_df.loc[positions_df["id"]==2]

test['cp_0'] = list(zip(player_1_pos.frame, player_1_pos.x, player_1_pos.y))
test['cp_1'] = list(zip(player_2_pos.frame, player_2_pos.x, player_2_pos.y))
print('cp_0', test['cp_0'])

cp_0 = pd.DataFrame(test['cp_0'], columns=['frame', 'x', 'y'])
cp_1 = pd.DataFrame(test['cp_1'], columns=['frame', 'x', 'y'])
# print(len(test['cp_0']))
# print(len(test['cp_1']))

# print(cp_0)
# print(cp_1)

df3 = pd.merge(cp_0, cp_1, on=['frame'])
# print(df3)
df3.to_csv("test.csv")
# newFrame = pd.DataFrame(test)
# print(newFrame)

newFrame['coor_bev_0'] = newFrame['cp_0'].apply(lambda x: player_coor(x, M))
newFrame['coor_bev_1'] = newFrame['cp_1'].apply(lambda x: player_coor(x, M))
positions_0 = list(newFrame['coor_bev_0'])
positions_1 = list(newFrame['coor_bev_1'])

# print(positions_0)


output_video_path = './outputs/output_bird_eye_view.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# better if the fps is the same as the one of the original video
fps = 60

output_video = cv2.VideoWriter(output_video_path, fourcc, fps,
							  (output_width, output_height))

# define a court to trace the player's path
court_base = BirdEyeViewCourt(output_width, pad)

i = 0
while (True):

    if len(positions_0) == i:
        break

    # copy instance in order not to have an inheritance
    court = deepcopy(court_base)

    # players positions at each frame
    court.add_player(positions_0[i], 0,
    				(255, 0, 0), (0, 0, 0))
    court.add_player(positions_1[i], 1,
    				(38, 19, 15), (0, 0, 0))

    # players positions at each frame added to the path
    court_base.add_path_player(positions_0[i])
    court_base.add_path_player(positions_1[i])

    output_video.write(court.court)
    i += 1

output_video.release()

