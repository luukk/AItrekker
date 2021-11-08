import numpy as np
import math
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation
from birdeyeview.bird_eye_view import *
from copy import deepcopy
import plotly.express as px
from scipy.signal import argrelextrema
import re
from os import walk

def to_tuples(df):
    new_df = pd.DataFrame(df['frame'])
    new_column_names = [f"{column[:-2]}" for column in df if re.search(r'.+?(?=_x)', column) ]
    pair_columns = pair([col for col in df.columns for column_name in new_column_names if column_name in col])

    i = 0
    for x, y in pair_columns:
        new_df[f"{new_column_names[i]}"] = list(zip(df[x], df[y]))
        i += 1

    return new_df


def pair(iterable):
    a = iter(iterable)
    return zip(a, a)


def format_columns(df):
    formatted_df = pd.DataFrame(df['frame'])
    transformed_columns = [column for column in df if '_T' in column]
    
    for t_column in transformed_columns:
        formatted_df[[f"{t_column[:-2]}_x", f"{t_column[:-2]}_y"]] = pd.DataFrame(df[t_column].tolist())
    
    return formatted_df.astype({ 'frame': int })



# animation output dimension
output_width = 1000
pad = 0.22
output_height = int(output_width * (1 - pad) * 2 * (1 + pad))

# coordinates of the 4 corners of the court

# image_pts = np.array([(561, 230), 
# 	                  (1233, 230),
# 	                  (1575, 867),
# 	                  (216, 867)]).reshape(4, 2)

image_pts = np.array([(561, 259),
                        (1261, 259),
                        (1618, 913),
                        (216, 913)]).reshape(4, 2)
bev_pts = np.array(court_coor(output_width, pad)).reshape(4, 2)

# homography matrix to go from real world image to bird eye view
M = transition_matrix(image_pts, bev_pts)

f = []
for (dirpath, dirnames, filenames) in walk("csv/merge"):
    f.extend(filenames)
    break

print(f)
# path = "1.csv"
for path in f:
	# print("getting points from file: ", path)
	df = pd.read_csv('csv/merge/'+path+'')
	df = pd.DataFrame(df, columns=['frame', 'ball_x', 'ball_y'])
	df = df.rename(columns={'ball_x': 'x', 'ball_y': 'y'})
	df = df[df['y'].notna()]
	df = df[df['x'].notna()]
	# df = df.loc[df["class_name"]=='ball']
	# df[['ball_x', 'ball_y']]
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

	data = {'frames': df.frame, 'x': x, 'y':y}
	filtered_frames = pd.DataFrame(data,columns=['frames', 'x', 'y'])
	filtered_frames = filtered_frames.drop(filtered_frames[filtered_frames.x < -400].index)
	filtered_frames = filtered_frames.drop(filtered_frames[filtered_frames.y < -500].index)

	filtered_frames['frame_nr_dif'] = filtered_frames['frames'].diff()
	filtered_frames['frame_nr_dif'] = pd.to_numeric(filtered_frames['frame_nr_dif'])
	cutoffs = filtered_frames.loc[filtered_frames['frame_nr_dif']>10]

	test = filtered_frames.loc[filtered_frames['frames'] == 29].size
	reindex = filtered_frames.reset_index(drop=True)

	cutoffs = reindex.loc[reindex['frame_nr_dif']>9]
	# print(cutoffs, cutoffs.index)

	all_critical_points = []

	output_frame = {'frame': [], 'x': [], 'y': []}
	found_frames = []

	for id, cutoff in enumerate(cutoffs.index):
		if id > 0:
			df = filtered_frames.iloc[int(cutoffs.index[id-1]): int(cutoff)]
			first_frame = df['frames'].iloc[0]
			last_frame = df['frames'].iloc[-1]
			bounds = [int(first_frame), int(last_frame)]
		else:
			df = filtered_frames.iloc[:int(cutoff)]
			first_frame = df['frames'].iloc[0]
			last_frame = df['frames'].iloc[-1]
			bounds = [int(first_frame), int(last_frame)]

		""" trend y-to-frames """
		trend = np.polyfit(df.frames, df.y, 10)
		trendpoly = np.poly1d(trend) 

		""" trend x-to-frames """
		trendx = np.polyfit(df.frames, df.x, 10)
		trendpolyx = np.poly1d(trendx) 

		plt.plot(df.frames, df.y, 'o')	
		plt.plot(list(range(bounds[0], bounds[1])),trendpoly(list(range(bounds[0], bounds[1]))), color='green')

		crit_x_points = [(int(y.real), int(trendpoly(y.real))) for y in trendpoly.deriv().r if y.imag == 0 and bounds[0] < y.real < bounds[1]]

		for point in crit_x_points:
			plt.plot(point[0], point[1], 'ro')	

		for critical_point in crit_x_points:
			x_point = int(trendpolyx(critical_point[0]))

			found_frames.append(critical_point[0])

			output_frame['frame'].append(critical_point[0])
			output_frame['x'].append(x_point)
			output_frame['y'].append(critical_point[1])

			all_critical_points.append((x_point, critical_point[1])) 
		# plt.show()

	# print("critical y points: ", all_critical_points)
	# print("critical x points: ", all_critical_x_points)

	pd.DataFrame(output_frame,columns=['frame', 'x', 'y']).to_csv("csv/out/"+path)


""" Merge CSVs """
print("merging all game csv's into one csv...")
# merge ball bounces with transformed tennis data
tennis_data_base_path = "csv/transformed/"
bounce_data_base_path = "csv/out/"
merged_transformed_write_path = "csv/transformed_new/"
for path in f:
	tennis_df = pd.read_csv(tennis_data_base_path + path)
	bounce_df = pd.read_csv(bounce_data_base_path + path)
	bounce_df = pd.DataFrame(bounce_df, columns=['frame', 'x', 'y'])
	bounce_df = bounce_df.rename(columns={'x': 'bounce_x', 'y': 'bounce_y'})

	merged_transformed_data = pd.merge(tennis_df, bounce_df, on=['frame'], how='left')
	merged_transformed_data.to_csv(merged_transformed_write_path + path)
""" Merge CSVs """


fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
ax1.plot(filtered_frames.frames,filtered_frames.x, 'o')
ax2.plot(filtered_frames.frames,filtered_frames.y, 'o')
# for point in all_critical_x_points:
# 	ax1.plot(point[0], point[1], 'ro')
for point in all_critical_points:
	ax2.plot(point[0], point[1], 'ro')

# plt.show()

for path in f:
	formatted_df = pd.read_csv(merged_transformed_write_path + path)
	bird_eye_coords = [list(to_tuples(formatted_df).iloc[:, (n + 1)]) for n in range(4)]    
# print("bird eye coords")
# print(bird_eye_coords)

	output_video_path = './outputs/final/video_'+path+'.avi'
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# better if the fps is the same as the one of the original video
	fps = 30

	output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

	# define a court to trace the player's path
	court_base = BirdEyeViewCourt(output_width, pad)

	colors = [
		(255, 20, 20),  # player1
		(20, 20, 255),  # player2
		(220, 253, 80), # ball
		(0,0,0)			# bounce
	]

	print("start generating video", path)
	i = 0
	show_balls = []
	while True:
		if len(bird_eye_coords[0]) == i:
			break
		# if filtered_frames['frames'].max() == i:
		# 	break

		court = deepcopy(court_base)
		
		# players positions at each frame
		# skips NaN entries
		# player1
		if not math.isnan(bird_eye_coords[0][i][0]):
				player_1_coords = tuple(map(int, bird_eye_coords[0][i]))
				court.add_player(player_1_coords, 0, colors[0], colors[0])
				court_base.add_path_player(player_1_coords, colors[0])
		# player2
		if not math.isnan(bird_eye_coords[1][i][0]):
			player_2_coords = tuple(map(int, bird_eye_coords[1][i]))
			court.add_player(player_2_coords, 1, colors[1], colors[1])
			court_base.add_path_player(player_2_coords, colors[1])
		# ball
		if not math.isnan(bird_eye_coords[2][i][0]):
			ball_coords = tuple(map(int, bird_eye_coords[2][i]))
			court.add_player(ball_coords, 1, colors[2], colors[2])
			court_base.add_path_player(ball_coords, colors[2])
		#bounce
		if not math.isnan(bird_eye_coords[3][i][0]):
			bounce_coords = tuple(map(int, bird_eye_coords[3][i]))
			court.add_player(bounce_coords, 1, colors[3], colors[3])
			court_base.add_path_player(bounce_coords, colors[3], True)
		
		# if i in found_frames:
		# 	show_balls.append(i)
		# 	print(show_balls)

		# for nr in show_balls:
		# 	index = found_frames.index(nr)
		# 	court.add_player(all_critical_points[index], 0, (255,0,0), (0,0,0))
		
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