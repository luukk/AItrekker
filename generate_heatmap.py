import os
import re
import shutil
from copy import deepcopy
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS

from birdeyeview.bird_eye_view import *

flags.DEFINE_string('input', None, 'Path to data folder')
flags.DEFINE_string('output', None, 'Path to write image to')


def main(_argv):
    if FLAGS.input:
        folder_path = f'{FLAGS.input}'
    else:
        folder_path = './output/transformed'

    if FLAGS.output:
        write_path = f'{FLAGS.output}'
    else:
        write_path = './output/heatmap'

    file_paths = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    files_dfs = [convert_dtypes(pd.read_csv(f'{folder_path}/{file_path}'), 'Int64') for file_path in file_paths]
    tupleized_dfs = [df.iloc[:, 1:3] for df in list(map(to_tuples, files_dfs))]

    players_coordinates = dict.fromkeys(tupleized_dfs[0].columns)
    for player in players_coordinates.keys():
        coordinates = []
        for df in tupleized_dfs:
            coordinates += list(df[player])
        players_coordinates[player] = coordinates

    pad = 0.22
    output_width = 1000
    output_height = int(output_width * (1 - pad) * 2 * (1 + pad))
    court_base = BirdEyeViewCourt(output_width, pad)

    positions_canvases = { 
        player: generate_blank_canvas(output_height, output_width)
        for player in players_coordinates.keys()
    }

    for player, img in positions_canvases.items():
        cv2.imwrite(f'{write_path}/{player}_temp.png', img)

    positions_canvases = { 
        player: cv2.imread(f'{write_path}/{player}_temp.png')
        for player in players_coordinates.keys()
    }

    for player in players_coordinates.keys():
        for x, y in players_coordinates[player]:
            if type(x) == np.int64:
                positions_canvases[player][y][x] = increment_color_values(positions_canvases[player][y][x])

    spread_heat_imgs = {
        player: cv2.GaussianBlur(positions_canvases[player], (31, 31), 0)
        for player in players_coordinates.keys()
    }

    normalized_spread_imgs = dict.fromkeys(players_coordinates.keys())
    for player in players_coordinates.keys():
        img_norm = None
        img_norm = cv2.normalize(spread_heat_imgs[player], img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        normalized_spread_imgs[player] = img_norm

    for player, img in normalized_spread_imgs.items():
        cv2.imwrite(f'{write_path}/{player}_BW.png', img)

    heatmap_imgs = {
        player: cv2.applyColorMap(normalized_spread_imgs[player], cv2.COLORMAP_JET)
        for player in players_coordinates.keys()
    }

    superimposed_imgs = {
        player: cv2.addWeighted(heatmap_imgs[player], 0.9, court_base.court, 0.2, 0)
        for player in players_coordinates.keys()
    }
    
    for player, img in superimposed_imgs.items():
        cv2.imwrite(f'{write_path}/{player}.png', img)

    clean_temp_files(write_path)


def generate_blank_canvas(height, width):
    return cv2.rectangle(
        np.zeros((height, width, 4)).astype(np.uint8),
        (0, 0),
        (height, width),
        (0, 0, 0),
        -1
    )


def increment_color_values(iterable):
    return np.array([e + 5 if i < 3 else 255 for i, e in enumerate(iterable)]).astype(np.uint8)


def convert_dtypes(df, type):
    for column in df.columns:
        df[column] = df[column].astype(type, copy=False)
    return df


def to_tuples(df):
    new_df = pd.DataFrame(df['frame'])
    new_column_names = [f'{column[:-2]}' for column in df if re.search(r'.+?(?=_x)', column) ]
    pair_columns = pair([col for col in df.columns for column_name in new_column_names if column_name in col])

    i = 0
    for x, y in pair_columns:
        new_df[f'{new_column_names[i]}'] = list(zip(df[x], df[y]))
        i += 1

    return new_df


def pair(iterable):
    a = iter(iterable)
    return zip(a, a)


def clean_temp_files(output_path):
    temp_files = [f for f in listdir(output_path) if isfile(join(output_path, f)) and '_temp' in f]
    for temp_file in temp_files:
        os.remove(f'{output_path}/{temp_file}')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
