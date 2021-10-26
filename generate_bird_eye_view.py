import pickle
import re
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS

from birdeyeview.bird_eye_view import *

flags.DEFINE_string('input', None, 'Path to data file to read from; e.g. \'.csv\'')
flags.DEFINE_string('write', None, 'Path to write csv; e.g. \'.csv\'')
flags.DEFINE_string('output', None, 'Path to output; e.g. \'output.avi\'')
# python generate_bird_eye_view.py --input ./output/merged/1.csv --write_to ./output/transformed/1.csv --output ./output/2d/1.avi
# python generate_bird_eye_view.py --input 1


def main(_argv):
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

    # df = pd.read_csv(f"{FLAGS.input}")
    df = pd.read_csv(f"./output/merged/{FLAGS.input}.csv") 
    coordinates_df = to_tuples(df)

    for column in coordinates_df:
        if column == 'frame': continue

        coordinates_df[f"{column}_T"] = coordinates_df[column].apply(lambda x: player_coor(x, M))

    # if FLAGS.write:
    #     write_to_file(coordinates_df)
    write_to_file(coordinates_df) ################### TEMPORARY



    # positions_0 = list(newFrame['coor_bev_0'])
    # positions_1 = list(newFrame['coor_bev_1'])

    # if FLAGS.output:
    #     output_video_path = f"{FLAGS.output}"
    # else:
    #     output_video_path = 'output.avi'

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # # better if the fps is the same as the one of the original video
    # fps = 60

    # output_video = cv2.VideoWriter(output_video_path, fourcc, fps,
    #                             (output_width, output_height))

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
    #                     (255, 0, 0), (0, 0, 0))
    #     court.add_player(positions_1[i], 1,
    #                     (38, 19, 15), (0, 0, 0))

    #     # players positions at each frame added to the path
    #     court_base.add_path_player(positions_0[i])
    #     court_base.add_path_player(positions_1[i])

    #     output_video.write(court.court)
    #     i += 1

    # output_video.release()


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


def write_to_file(df):
    write_df = pd.DataFrame(df['frame'])
    transformed_columns = [column for column in df if '_T' in column]
    
    for t_column in transformed_columns:
        write_df[[f"{t_column[:-2]}_x", f"{t_column[:-2]}_y"]] = pd.DataFrame(df[t_column].tolist())
    
    write_df = write_df.astype({ 'frame': int })
    # write_df.to_csv(f"{FLAGS.write}", index=False)
    write_df.to_csv(f'./output/transformed/{FLAGS.input}.csv', index=False) ################### TEMPORARY


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
