import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('input', None, 'Data file to read from; e.g. \'.csv\'')
flags.DEFINE_string('output', None, 'Data file to write to; e.g. \'.csv\'')


def main(_argv):
    df = pd.read_csv(f"{FLAGS.input}")

    p1, p2 = get_player_positions(df)
    # test = p1['x'].interpolate(method='linear')
    # test2 = p2['x'].interpolate(method='linear')
    test2 = interpolate_missing_frames(p2)

    if FLAGS.output:
        p2.to_csv('./output/corrected/nan.csv')
        # for player in player_positions:
        test2.to_csv(f"{FLAGS.output}")


def interpolate_missing_frames(df):
    breakpoint()


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


def get_player_positions(df):
    p1 = df[df['id'] == '1']
    p2 = df[df['id'] == '2']
    p1_dict = p1.to_dict(orient='index')
    p2_dict = p2.to_dict(orient='index')

    max_frames = max(p1['frame'].iloc[-1], p2['frame'].iloc[-1])
    p1 = fill_missing_frames(p1_dict, p1['frame'][0], max_frames, '1')
    p2 = fill_missing_frames(p2_dict, p2['frame'][1], max_frames, '2')

    return [p1, p2]


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
