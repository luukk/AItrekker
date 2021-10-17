import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation
from birdeyeview.bird_eye_view import *
from copy import deepcopy
import plotly.express as px
from scipy.signal import argrelextrema

# animation output dimension
output_width = 1920
pad = 0.22
output_height = int(output_width * (1 - pad) * 2 * (1 + pad))

# coordinates of the 4 corners of the court
image_pts = np.array([(561, 230), 
	                  (1233, 230),
	                  (1575, 867),
	                  (216, 867)]).reshape(4, 2)
bev_pts = np.array(court_coor(output_width, pad)).reshape(4, 2)

# homography matrix to go from real world image to bird eye view
M = transition_matrix(image_pts, bev_pts)


df = pd.read_csv('csv/tracking_players_4.csv')
df = df.loc[df["class_name"]=='ball']
df = df.drop(df[df.x < -400].index)

points = np.array([(1, 1), (2, 4), (3, 1), (9, 3)])
# # get x and y vectors
# x = points[:,0]
# y = points[:,1]
df['xy'] = list(zip(df.x, df.y))
test = df['xy'].apply(lambda x: player_coor(x, M))

positions_0 = list(test)

x =  [i[0] for i in positions_0]
y =  [i[1] for i in positions_0]

data = {'frames': df.id, 'x': x, 'y':y}
yee = pd.DataFrame(data,columns=['frames', 'x', 'y'])
yee = yee.drop(yee[yee.x < -400].index)
yee = yee.drop(yee[yee.y < -500].index)

yee.to_csv("output_yee.csv")
animation_frame="frame"

yee['frame_nr_dif'] = yee['frames'].diff()
yee['frame_nr_dif'] = pd.to_numeric(yee['frame_nr_dif'])
cutoffs = yee.loc[yee['frame_nr_dif']>10]

test = yee.loc[yee['frames'] == 29].size
reindex = yee.reset_index(drop=True)

cutoffs = reindex.loc[reindex['frame_nr_dif']>9]
print(cutoffs, cutoffs.index)

all_critical_points = []

output_frame = {'frame': [], 'x': [], 'y': []}
found_frames = []

for id, cutoff in enumerate(cutoffs.index):
	if id > 0:
		df = yee.iloc[int(cutoffs.index[id-1]): int(cutoff)]
		first_frame = df['frames'].iloc[0]
		last_frame = df['frames'].iloc[-1]
		bounds = [first_frame, last_frame]
	else:
		df = yee.iloc[:int(cutoff)]
		first_frame = df['frames'].iloc[0]
		last_frame = df['frames'].iloc[-1]
		bounds = [first_frame, last_frame]

	""" trend y-to-frames """
	trend = np.polyfit(df.frames, df.y, 10)
	trendpoly = np.poly1d(trend) 

	""" trend x-to-frames """
	trendx = np.polyfit(df.frames, df.x, 10)
	trendpolyx = np.poly1d(trendx) 

	plt.plot(df.frames, df.y, 'o')	
	plt.plot(list(range(bounds[0], bounds[1])),trendpoly(list(range(bounds[0], bounds[1]))), color='green')

	crit_x_points = [(int(y.real), int(trendpoly(y.real))) for y in trendpoly.deriv().r if y.imag == 0 and bounds[0] < y.real < bounds[1]]
	print(crit_x_points)

	for point in crit_x_points:
		plt.plot(point[0], point[1], 'ro')	

	for critical_point in crit_x_points:
		x_point = int(trendpolyx(critical_point[0]))

		found_frames.append(critical_point[0])

		output_frame['frame'].append(critical_point[0])
		output_frame['x'].append(x_point)
		output_frame['y'].append(critical_point[1])

		all_critical_points.append((x_point, critical_point[1])) 
	plt.show()

# print("critical y points: ", all_critical_points)
# print("critical x points: ", all_critical_x_points)
print(output_frame)


pd.DataFrame(output_frame,columns=['frame', 'x', 'y']).to_csv("ball_hit_locations_5.csv")


fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
ax1.plot(yee.frames,yee.x, 'o')
ax2.plot(yee.frames,yee.y, 'o')
# for point in all_critical_x_points:
# 	ax1.plot(point[0], point[1], 'ro')
for point in all_critical_points:
	ax2.plot(point[0], point[1], 'ro')

# plt.show()

output_video_path = './outputs/output_bird_eye_view_hit_5.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# better if the fps is the same as the one of the original video
fps = 30

output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# define a court to trace the player's path
court_base = BirdEyeViewCourt(output_width, pad)


i = 0
show_balls = []
while True:
	if yee['frames'].max() == i:
		break

	court = deepcopy(court_base)
	
	if i in found_frames:
		show_balls.append(i)
		print(show_balls)

	for nr in show_balls:
		index = found_frames.index(nr)
		court.add_player(all_critical_points[index], 0, (255,0,0), (0,0,0))
	
	output_video.write(court.court)
	i+=1

output_video.release()

# video1 = cv2.VideoCapture('./outputs/output_bird_eye_view_hit_5.avi')
# video2 = cv2.VideoCapture('./outputs/demo-5.avi')

# while True:
# 	ret1, frame1 = video1.read()
# 	ret2, frame2 = video2.read()

# 	if ret1==False or ret2==False:
# 		break
# 	frame1 = cv2.resize(frame1, (960, 540))
# 	frame2 = cv2.resize(frame2, (960, 540))

# 	dst = np.concatenate((frame1, frame2))

# 	cv2.imshow('dst',dst)
# 	key =cv2.waitKey(1)
# 	if key == ord('q'):
# 		break

# cv2.destroyAllWindows()
