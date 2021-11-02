from os import listdir
from os.path import isfile, join
from pathlib import Path

import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('input', None, 'Path to data folder')


def main(_argv):
    if FLAGS.input:
        path = f"{FLAGS.input}"
    else:
        path = './output/transformed'

    file_paths = [f for f in listdir(path) if isfile(join(path, f))]
    files_dfs = [pd.read_csv(f'{path}/{file_path}') for file_path in file_paths]
    breakpoint()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
