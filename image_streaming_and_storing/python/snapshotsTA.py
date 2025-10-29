#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 SICK AG, Waldkirch
#
# SPDX-License-Identifier: Unlicense

import argparse
import os
import time
from time import perf_counter as pc
import numpy as np
import pickle

from base.python.Control import Control
from base.python.Stream import Streaming
from base.python.Streaming import Data
from base.python.Streaming.BlobServerConfiguration import BlobClientConfig
from shared.python.data_processing import processSensorData
from shared.python.devices_config import get_device_config
from base.python.Usertypes import FrontendMode


def convertToPointCloudOptimized(distData: list, myCamParams: list):
    """
    This function converts 2D image data to a 3D point cloud.

    Parameters:
    distData (list): The distance data from the sensor, as a list.
    cnfiData (list): The confidence data from the sensor, as a list.
    myCamParams (list): The camera parameters, as a list.
    isStereo (bool): A flag indicating whether the camera is a stereo camera.

    Returns:
    wCoordinates (numpy.ndarray): A 3D numpy array of shape (camera height, camera width, 3). Each triplet at position [y,x,:] represents the 3D coordinates (in millimeters) of the point at (y_pixel,x_pixel) in the image. So, x_mms, y_mms, z_mms.

    Usage:
    To use the returned point cloud, you can do the following:

    cloud = convertToPointCloud(distData, cnfiData, myCamParams, isStereo)
    x_MMS, y_MMS, z_MMS = cloud[y_pixel, x_pixel, :]

    Note:
    This function overwrites the camera mounting setting parameters from SOPAS. These will always be ignored.
    """

    # wCoordinates = []

    # the distance from the sensor to the origin of the camera coordinate system in z
    SENSOR_TO_ORIGIN_DIST = myCamParams.cam2worldMatrix[11]

    # the orientation matrix of the camera, ignoring all sopas/previous configurations!!!!!!!!!!!!!!!!
    m_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, SENSOR_TO_ORIGIN_DIST],
        [0, 0, 0, 1]
    ])

    shape = (4, 4)
    m_c2w.shape = shape
    distData = np.asarray(list(distData)).reshape(
        myCamParams.height, myCamParams.width)
    # cnfiData = np.asarray(cnfiData).reshape(
    #     myCamParams.height, myCamParams.width)


    # return wCoordinates
    cols = np.arange(0, myCamParams.width)
    rows = np.arange(0, myCamParams.height)
    xp = (cols - myCamParams.cx) / myCamParams.fx
    yp = (rows - myCamParams.cy) / myCamParams.fy
    xp = xp[:, np.newaxis]
    yp = yp[np.newaxis, :]

    xc = distData * xp.T
    yc = yp.T * distData

    zc = distData

    xw = (m_c2w[0, 3] + zc * m_c2w[0, 2] +
          yc * m_c2w[0, 1] + xc * m_c2w[0, 0])
    yw = (m_c2w[1, 3] + zc * m_c2w[1, 2] +
          yc * m_c2w[1, 1] + xc * m_c2w[1, 0])
    zw = (m_c2w[2, 3] + zc * m_c2w[2, 2] +
          yc * m_c2w[2, 1] + xc * m_c2w[2, 0])

    wCoordinates = np.stack([xw, yw, zw], axis=-1)
    # cloud_data_like_sopas = wCoordinates.reshape(
    #     (myCamParams.height, myCamParams.width, 3))
    '''
    offset the entire cordinate system so that the middle pixel is at (0,0,z) ->
    i want to define the system so that the technical drawing, point 7 in here: https://www.sick.com/il/en/catalog/products/machine-vision-and-identification/machine-vision/visionary-t-mini/v3s105-1aaaaaa/p/p665983?tab=detail
    is at (0,0,z) in the world cordinate system.
    that point is defined in pixel coordinates as: (myCamParams.width // 2, myCamParams.height // 2). 
    so range_data[myCamParams.height // 2, myCamParams.width // 2] is the distance from sensor of that point.
    and cloud_data[myCamParams.height // 2, myCamParams.width // 2,:] is the 3D coordinates of that point.
    '''
    # cloud_data = cloud_data_like_sopas.copy()
    # sensor_center_x_y_values = (cloud_data_like_sopas[myCamParams.height // 2, myCamParams.width //
    #                             2, :2] + cloud_data_like_sopas[(-1+myCamParams.height) // 2, (-1+myCamParams.width) // 2, :2])/2
    # making the middle pixel be the origin of the coordinates system (x=0,y=0) and each point is relative to it
    # cloud_data[:, :, :2] = cloud_data_like_sopas[:, :, :2] - \
    #     sensor_center_x_y_values
    # where cnfiData is 0, so set it to a numpy array of 0,0,0
    # cloud_data[cnfiData != 0] = np.array([0, 0, 0])

    return wCoordinates


def save2pointcloud(frame_number, depth_map, camera_params, pcl_dir):
    # if sensor_data.hasDepthMap:
    frame_number = frame_number
    # print("Data contains depth map data")

    if write_files:
        # print("=== Write PNG file: Frame number: {}".format(frame_number))
        # writeFrame(device_type, sensor_data,
        #            os.path.join(img_dir, output_prefix))
        # print("=== Converting image to pointcloud")

        # Non optimized
        # start_time = time.time()
        # world_coordinates, dist_data = convertToPointCloud(
        #     sensor_data.depthmap.distance,
        #     sensor_data.depthmap.intensity,
        #     # sensor_data.depthmap.confidence,
        #     sensor_data.cameraParams,
        #     sensor_data.xmlParser.stereo
        # )
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"convertToPointCloud took: {execution_time:.3}s")

        # Optimized
        # start_time = time.time()
        point_cloud = convertToPointCloudOptimized(
            depth_map,
            camera_params,
        )
        end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"convertToPointCloudOptimized took: {execution_time:.3}s")

        pickle.dump(
            point_cloud, open(f"{pcl_dir}/world_coordinates{frame_number}.pickle", 'wb')
        )


def runSnapshotsDemo(device_control: Control, ip_address: str, transport_protocol: str, receiver_ip: str,
                     streaming_port: int, device_type: str,  number_frames: int, output_prefix: str,
                     poll_period_ms: int, write_files: bool):
    pcl_dir = None
    img_dir = None
    if write_files:
        # directory to save the output in
        pcl_dir = 'VisionaryToPointCloud'
        img_dir = 'VisionaryImages'
        os.makedirs(pcl_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

    # streaming settings:
    streaming_settings = BlobClientConfig(device_control)
    streaming_device = None

    if transport_protocol == "TCP":
        # configure the data stream, the methods immediately write the setting to the device
        # set protocol and device port
        streaming_settings.setTransportProtocol(
            streaming_settings.PROTOCOL_TCP)
        streaming_settings.setBlobTcpPort(streaming_port)
        # start streaming
        streaming_device = Streaming(ip_address, streaming_port)
        streaming_device.openStream()

    elif transport_protocol == "UDP":
        # settings
        streaming_settings.setTransportProtocol(
            streaming_settings.PROTOCOL_UDP)  # UDP
        streaming_settings.setBlobUdpReceiverPort(streaming_port)
        streaming_settings.setBlobUdpReceiverIP(receiver_ip)
        streaming_settings.setBlobUdpControlPort(streaming_port)
        streaming_settings.setBlobUdpMaxPacketSize(1024)
        streaming_settings.setBlobUdpIdleTimeBetweenPackets(
            10)  # in milliseconds
        streaming_settings.setBlobUdpHeartbeatInterval(0)
        streaming_settings.setBlobUdpHeaderEnabled(True)
        streaming_settings.setBlobUdpFecEnabled(
            False)  # forward error correction
        streaming_settings.setBlobUdpAutoTransmit(True)
        streaming_device = Streaming(
            ip_address, streaming_port, protocol=transport_protocol)
        streaming_device.openStream((receiver_ip, streaming_port))
    
    # tag::stop_frontend[]
    # Stop image acquisition (works always, also when already stopped)
    device_control.setFrontendMode(FrontendMode.Stopped)
    # end::stop_frontend[]

    # logout after settings have been done
    device_control.logout()

    # tag::avoid_overrun[]
    poll_period_span = poll_period_ms / 1000.0  # Convert milliseconds to seconds
    last_snap_time = time.time()
    # end::avoid_overrun[]

    sensor_data = Data.Data()

    # trigger dummy snapshot acquistion to restart frontend 
    # (a stopped frontend needs to warm up for 16 frames to achieve specified TOF precision, these frames will be dropped internally)
    device_control.singleStep()
    streaming_device.getFrame()

    # acquire a single snapshot
    for i in range(number_frames):
        t0 = pc()
        # tag::avoid_overrun[]
        # make sure we don't overrun the device
        # (otherwise snapshot requests would be dropped by the device)
        time_since_last_snap = time.time() - last_snap_time

        if time_since_last_snap < poll_period_span:
            time_to_wait = poll_period_span - time_since_last_snap
            time.sleep(time_to_wait)
        # end::avoid_overrun[]

        # tag::acquire_snapshots[]
        # now we are not too fast and can trigger a snapshot
        last_snap_time = time.time()
        device_control.singleStep()
        streaming_device.getFrame()
        whole_frame = streaming_device.frame
        sensor_data.read(whole_frame, convertToMM=False)

        save2pointcloud(
            sensor_data.depthmap.frameNumber, sensor_data.depthmap.distance, sensor_data.cameraParams, pcl_dir
        )
        # processSensorData(sensor_data, device_type,
        #                   img_dir, output_prefix, pcl_dir, write_files)

        # end::acquire_snapshots[]
        print(f'{i}-{pc() - t0:.3f}s')

    # tag::close_streaming[]
    device_control.login(Control.USERLEVEL_AUTH_CLIENT, 'CLIENT')
    streaming_device.closeStream()
    if transport_protocol == "UDP":
        # restoring back to TCP mode
        streaming_settings.setTransportProtocol(
            streaming_settings.PROTOCOL_TCP)
    streaming_settings.setBlobTcpPort(2114)
    # end::close_streaming[]

    # tag::logout_and_close[]
    # Reset the image acquisition to default mode
    device_control.setFrontendMode(FrontendMode.Continuous)
    device_control.logout()
    device_control.close()
    # end::logout_and_close[]
    print("Logout and close")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script demonstrates how to read system health variables.")
    parser.add_argument('-i', '--ipAddress', required=False, type=str,
                        default="192.168.1.10", help="The ip address of the device.")
    parser.add_argument('-t', '--transport_protocol', required=False, choices=['TCP', 'UDP'],
                        default="TCP", help="The transport protocol.")
    parser.add_argument('-r', '--receiver_ip', required=False, type=str,
                        default="192.168.1.2", help="The ip address of the receiving PC (UDP only).")
    parser.add_argument('-d', '--device_type', required=False, type=str,
                        default="Visionary-T Mini", choices=["Visionary-S", "Visionary-T Mini"],
                        help="Visionary product type.")
    parser.add_argument('-s', '--streaming_port', required=False, type=int,
                        default=2114, help="The port of the data channel.")
    parser.add_argument('-p', '--poll_period_ms', required=False, type=int,
                        default=500, help="Poll period for snapshot in ms.")
    parser.add_argument('-n', '--count', required=False, type=int, default=10,
                        help="Acquire number frames and stop.")
    parser.add_argument('-o', "--output_prefix", required=False, type=str, default="",
                        help="prefix for the output files.")
    parser.add_argument('-w', "--write_files", required=False, type=str, choices=["True", "False"],
                        default="True", help="Write the files to storage if True.")
    args = parser.parse_args()

    cola_protocol, control_port, _ = get_device_config(args.device_type)

    write_files = True if args.write_files == "True" else False

    runSnapshotsDemo(args.ipAddress, args.transport_protocol, args.receiver_ip,
                     cola_protocol, control_port, args.streaming_port, args.device_type,
                     args.count, args.output_prefix, args.poll_period_ms, write_files)
