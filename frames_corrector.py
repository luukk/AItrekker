import math
import re
from functools import reduce

import numpy as np
import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('input', None, 'Data file to read from; e.g. \'.csv\'')
flags.DEFINE_string('output', None, 'Data file to write to; e.g. \'.csv\'')
flags.DEFINE_string('near', None, 'Label for player closer to the camera')
flags.DEFINE_string('far', None, 'Label for player further from the camera')
# python data_processor.py --input ./output/data/1.csv --output ./output/frames_corrected/1.csv --near WOZ --far SHA


def main(_argv):
    df = pd.read_csv(f"{FLAGS.input}")

    merged_df = transform_to_columns(df)

    for column_name in merged_df:
        if re.search(r'_x|_y', column_name) and 'ball_' not in column_name:
            merged_df[f"{column_name}"].interpolate(method='index', inplace=True)
        else:
            continue

    merged_df = merged_df.astype({ 'frame': int })

    if FLAGS.output:
        merged_df.to_csv(f"{FLAGS.output}", index=False)
    else:
        input_file = re.findall(r'[^\/]+(?=\.)', FLAGS.input)[0]
        merged_df.to_csv(f"{input_file}.csv", index=False)


def fill_missing_frames(df, first_frame, last_frame, label):
    frame_set_range = range(first_frame, last_frame)
    dc = {}
    dcf = {}

    for k, v in df.items():
        dc[v['frame']] = {'idx': v['idx'], 'x': v['x'], 'y': v['y']}

    for i in frame_set_range:
        if(i not in dc):
            dc[i] = {'idx': -1, 'x': float('nan'), 'y': float('nan')}

    offset = 0
    current = 0
    for i in sorted(dc.keys()):
        if(dc[i]['idx'] == -1):
            offset += 1
        dcf[str(current)] = {'frame': i, f"{label}_x": dc[i]['x'], f"{label}_y": dc[i]['y']}
        current += 1

    return pd.DataFrame.from_dict(dcf).transpose()


def transform_to_columns(df):
    # because our class ids can only be the 2 players or the ball
    dfs = [df[df['id'] == f"{class_id}"].reset_index(drop=True) for class_id in df['id'].unique()]

    first_frame = min(dfs[0]['frame'][0], dfs[1]['frame'][0])
    last_frame = max(dfs[0]['frame'].iloc[-1], dfs[1]['frame'].iloc[-1])

    dfs = [df.to_dict(orient='index') for df in dfs]

    dfs = [
        fill_missing_frames(dfs[0], first_frame, last_frame, f"{FLAGS.near}"),
        fill_missing_frames(dfs[1], first_frame, last_frame, f"{FLAGS.far}"),
        fill_missing_frames(dfs[2], first_frame, last_frame, 'ball')
    ]

    return reduce(lambda df1, df2: pd.merge(df1, df2, on='frame'), dfs)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
