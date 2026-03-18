#!/usr/bin/env python3
"""Inspect LiDAR data in ROS2 bag file."""

import argparse
from pathlib import Path
from rosbags.rosbag2 import Reader as Reader2

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper


def inspect_lidar_bag(bag_path: str, lidar_topic: str = '/velodyne/points'):
    bag = Path(bag_path)

    if not bag.exists():
        print(f"Error: Bag path does not exist: {bag}")
        return

    print(f"Inspecting bag: {bag}")
    print(f"LiDAR topic: {lidar_topic}\n")

    bag_wrapper = Ros2BagWrapper(bag_path, None)
    typestore = bag_wrapper.get_typestore()
    num_msgs = bag_wrapper.get_topic_count(lidar_topic)
    print(f"Found {num_msgs} messages on topic '{lidar_topic}'\n")

    point_counts = []
    msg_count = 0

    with Reader2(str(bag)) as reader:
        connections = [x for x in reader.connections if x.topic == lidar_topic]

        if not connections:
            print(f"Topic '{lidar_topic}' not found.")
            print(f"Available topics: {sorted(set(x.topic for x in reader.connections))}")
            return

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

            num_points = len(msg.data) // msg.point_step
            point_counts.append(num_points)

            if msg_count == 0:
                print(f"First message:")
                print(f"  frame_id:    {msg.header.frame_id}")
                print(f"  timestamp:   {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
                print(f"  is_bigendian:{msg.is_bigendian}")
                print(f"  point_step:  {msg.point_step} bytes")
                print(f"  fields ({len(msg.fields)}):")
                for f in msg.fields:
                    dtype_names = {1:'INT8',2:'UINT8',3:'INT16',4:'UINT16',
                                   5:'INT32',6:'UINT32',7:'FLOAT32',8:'FLOAT64'}
                    print(f"    [{f.offset:2d}] {f.name:<12} {dtype_names.get(f.datatype, f'type={f.datatype}')} x{f.count}")
                print(f"  num_points:  {num_points}")
                print()

            msg_count += 1

    # Summary
    print(f"{'='*60}")
    print(f"SUMMARY  ({msg_count} messages)")
    print(f"{'='*60}")
    print(f"  min points : {min(point_counts)}")
    print(f"  max points : {max(point_counts)}")
    print(f"  mean points: {np.mean(point_counts):.1f}")
    print(f"  std points : {np.std(point_counts):.1f}")

    unique, counts = np.unique(point_counts, return_counts=True)
    print(f"\nPoint count distribution:")
    for n, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {n:6d} pts  x {c:5d} msgs  ({100.0*c/msg_count:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect LiDAR data in ROS2 bag")
    parser.add_argument('--bag_path', type=str, required=True, help="Path to ROS2 bag")
    parser.add_argument('--topic', type=str, default='/velodyne/points', help="LiDAR topic")
    args = parser.parse_args()
    inspect_lidar_bag(args.bag_path, args.topic)
