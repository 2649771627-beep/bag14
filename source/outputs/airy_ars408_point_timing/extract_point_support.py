"""Re-extract timing of the original board clusters from the two R01 bags."""

import argparse
import csv
from pathlib import Path
import sys

import extract_recordings as original


RUNS = {
    "near_constant": "right_airy_ars408_2to6_r01_near_constant_20261119_154314",
    "variable": "right_airy_ars408_2to6_r01_variable_20261119_155200",
}
PRECHECK = "right_airy_2to3_precheck_20261119_153250"


def extract_board_times(bags, output):
    background, empty_frames = original.empty_scene(bags / PRECHECK)
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("recording", "stamp_ns", "receive_ns", "x_m", "y_m", "z_m", "point_dt_s", "n_points"))
    for arm, name in RUNS.items():
        frames = tracked = 0
        with original.database(bags / name) as database:
            for receipt, raw in original.messages(database, original.LIDAR_TOPIC):
                cloud = original.deserialize_message(raw, original.PointCloud2)
                cluster = original.board_cluster(original.points_in_roi(cloud), background)
                frames += 1
                if cluster is not None:
                    tracked += 1
                    writer.writerow((arm, original.stamp_ns(cloud), receipt, *cluster))
        print(f"{arm}: {frames} clouds, {tracked} board clusters, {empty_frames} background frames", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", type=Path)
    extract_board_times(parser.parse_args().bags, sys.stdout)
