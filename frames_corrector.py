import math
import re

import numpy as np
import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('input', None, 'Data file to read from; e.g. \'.csv\'')
flags.DEFINE_string('output', None, 'Data file to write to; e.g. \'.csv\'')


def main(_argv):
    df = pd.read_csv(f"{FLAGS.input}")

    players = write_filled_frames(df)

    input_file = re.findall(r'[^\/]+(?=\.)', FLAGS.input)[0]
    players = [
        pd.read_csv(f"./output/frame_corrected/{input_file}_player_1.csv"),
        pd.read_csv(f"./output/frame_corrected/{input_file}_player_2.csv")
    ]

    for player in players:
        player['x'].interpolate(method='index', inplace=True)
        player['y'].interpolate(method='index', inplace=True)

    for idx, player_df in enumerate(players):
        if FLAGS.output:
            input_file = re.findall(r'[^\/]+(?=\.)', FLAGS.input)[0]
            player_df.to_csv(f"{FLAGS.output}/{input_file}_player_{idx + 1}.csv", index=False)
        else:
            player_df.to_csv(f"player_{idx + 1}.csv", index=False)


def fill_missing_frames(df, first_frame, max_frames, pid):
    frame_set_range = range(first_frame, max_frames)
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
        dcf[str(current)] = {'id': pid, 'frame': i, 'x': dc[i]['x'], 'y': dc[i]['y']}
        current += 1

    return pd.DataFrame.from_dict(dcf).transpose()


def write_filled_frames(df):
    p1 = df[df['id'] == '1']
    p2 = df[df['id'] == '2']
    p1_dict = p1.to_dict(orient='index')
    p2_dict = p2.to_dict(orient='index')

    max_frames = max(p1['frame'].iloc[-1], p2['frame'].iloc[-1])
    players = [fill_missing_frames(p1_dict, p1['frame'][0], max_frames, '1'), fill_missing_frames(p2_dict, p2['frame'][1], max_frames, '2')]

    for idx, player_df in enumerate(players):
        if FLAGS.output:
            input_file = re.findall(r'[^\/]+(?=\.)', FLAGS.input)[0]
            player_df.to_csv(f"./output/frame_corrected/{input_file}_player_{idx + 1}.csv", index=False)
        else:
            player_df.to_csv(f"player_{idx + 1}.csv", index=False)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
