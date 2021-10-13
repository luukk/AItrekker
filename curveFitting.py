import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation
from birdeyeview.bird_eye_view import *
from copy import deepcopy
import plotly.express as px


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


df = pd.read_csv('tracking_players_ball.csv')
df = df.loc[df["class_name"]=='ball']
df = df.drop(df[df.x < -400].index)
print(df)

points = np.array([(1, 1), (2, 4), (3, 1), (9, 3)])
print(points)
# # get x and y vectors
# x = points[:,0]
# y = points[:,1]
df['xy'] = list(zip(df.x, df.y))
test = df['xy'].apply(lambda x: player_coor(x, M))

positions_0 = list(test)
print(test)

x =  [i[0] for i in positions_0]
y =  [i[1] for i in positions_0]

data = {'x': x, 'y':y}
yee = pd.DataFrame(data,columns=['x', 'y'])
yee = yee.drop(yee[yee.x < -400].index)


# yee.to_csv("output_yee.csv")
print(yee)
animation_frame="frame"
# fig = px.scatter(yee, x="x", y="y", animation_frame="frame", size_max=45)
# fig.show()

# fig = py.figure(2)
# ax = py.axes(xlim=(0, 1), ylim=(0, 1))
# scat = ax.scatter([], [], s=60)

# def init():
#     scat.set_offsets([])
#     return scat,

# def animate(i):
#     scat.set_offsets([x[:i], y[:i]])
#     return scat,

# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x)+1, 
#                                interval=200, blit=False, repeat=False, cumulative=True)

trend = np.polyfit(yee.x[:20], yee.y[:20], 40)
plt.plot(yee.x[:20],yee.y[:20], 'o')
trendpoly = np.poly1d(trend) 
plt.plot(yee.x[:20],trendpoly(yee.x[:20]))
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
ax1.plot(yee.x[:40],yee.y[:40], 'o')
ax2.plot(yee.x[40:80],yee.y[40:80], 'o')
ax3.plot(yee.x[80:120],yee.y[80:120], 'o')
# ax4.plot(yee.x[150:200],yee.y[150:200], 'o')
# ax5.plot(yee.x[200:250],yee.y[200:250], 'o')
# ax6.plot(yee.x,yee.y, 'o')
plt.show()



# output_video_path = './outputs/output_bird_eye_view_ball.avi'
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # better if the fps is the same as the one of the original video
# fps = 30

# output_video = cv2.VideoWriter(output_video_path, fourcc, fps,
# 							  (output_width, output_height))

# # define a court to trace the player's path
# court_base = BirdEyeViewCourt(output_width, pad)

# i = 0
# while (True):

#     if len(positions_0) == i:
#         break

#     # copy instance in order not to have an inheritance
#     court = deepcopy(court_base)

#     # players positions at each frame
#     court.add_player(positions_0[i], 0,
#     				(255, 0, 0), (0, 0, 0))
    
#     # players positions at each frame added to the path
#     court_base.add_path_player(positions_0[i])
#     # court_base.add_path_player(positions_1[i])

#     output_video.write(court.court)
#     i += 1

# output_video.release()


# print(x,y)
# # calculate polynomial
# z = np.polyfit(x, y, 3)
# f = np.poly1d(z)

# # calculate new x's and y's
# x_new = np.linspace(x[0], x[-1], 50)
# y_new = f(x_new)

# plt.plot(x,y,'o', x_new, y_new)
# plt.xlim([x[0], x[-1] + 1 ])
# plt.show()