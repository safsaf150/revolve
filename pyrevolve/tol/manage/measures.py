import numpy as np
from collections import deque

from pyrevolve.SDF.math import Vector3, Quaternion
from pyrevolve.util import Time
import math

def velocity(robot_manager):
    """
    Returns the velocity over the maintained window
    :return:
    """
    return robot_manager._dist / robot_manager._time if robot_manager._time > 0 else 0

def displacement(robot_manager):
    """
    Returns a tuple of the displacement in both time and space
    between the first and last registered element in the speed
    window.
    :return: Tuple where the first item is a displacement vector
             and the second a `Time` instance.
    :rtype: tuple(Vector3, Time)
    """ 
    if robot_manager.last_position is None:
        return Vector3(0, 0, 0), Time()   
    return (
        robot_manager._positions[-1] - robot_manager._positions[0],
        robot_manager._times[-1] - robot_manager._times[0]
    )

def displacement_velocity(robot_manager):
    """
    Returns the displacement velocity, i.e. the velocity
    between the first and last recorded position of the
    robot in the speed window over a straight line,
    ignoring the path that was taken.
    :return:
    """
    dist, time = displacement(robot_manager)
    if time.is_zero():
        return 0.0
    return np.sqrt(dist.x**2 + dist.y**2) / float(time)

def displacement_velocity_hill(robot_manager):
    dist, time = displacement(robot_manager)
    if time.is_zero():
        return 0.0
    return dist.y / float(time)

def head_balance(robot_manager):
    """
    Returns the average rotation of teh head in the roll and pitch dimensions.
    :return:
    """
    roll = 0
    pitch = 0
    instants = len(robot_manager._orientations)
    for o in robot_manager._orientations:
        roll = roll + abs(o[0]) * 180 / math.pi
        pitch = pitch + abs(o[1]) * 180 / math.pi
    #  accumulated angles for each type of rotation
    #  divided by iterations * maximum angle * each type of rotation
    balance = (roll + pitch) / (instants * 180 * 2)
    # turns imbalance to balance
    balance = 1 - balance
    return balance


def sum_of_contacts(robot_manager):
    sum_of_contacts = 0
    for c in robot_manager._contacts:
        sum_of_contacts += c
    return sum_of_contacts

def logs_position_orientation(robot_manager, o, evaluation_time, robotid, path):

    with open(path+'/data_fullevolution/descriptors/positions_'+robotid+'.txt', "a+") as f:

        if robot_manager.second <= evaluation_time:
            robot_manager.avg_roll += robot_manager._orientations[o][0]
            robot_manager.avg_pitch += robot_manager._orientations[o][1]
            robot_manager.avg_yaw += robot_manager._orientations[o][2]
            robot_manager.avg_x += robot_manager._positions[o].x
            robot_manager.avg_y += robot_manager._positions[o].y
            robot_manager.avg_z += robot_manager._positions[o].z
            robot_manager.avg_roll = robot_manager.avg_roll/robot_manager.count_group
            robot_manager.avg_pitch = robot_manager.avg_pitch/robot_manager.count_group
            robot_manager.avg_yaw = robot_manager.avg_yaw/robot_manager.count_group
            robot_manager.avg_x = robot_manager.avg_x/robot_manager.count_group
            robot_manager.avg_y = robot_manager.avg_y/robot_manager.count_group
            robot_manager.avg_z = robot_manager.avg_z/robot_manager.count_group
            robot_manager.avg_roll = robot_manager.avg_roll * 180 / math.pi
            robot_manager.avg_pitch = robot_manager.avg_pitch * 180 / math.pi
            robot_manager.avg_yaw = robot_manager.avg_yaw * 180 / math.pi
            f.write(str(robot_manager.second) + ' '
                    + str(robot_manager.avg_roll) + ' '
                    + str(robot_manager.avg_pitch) + ' '
                    + str(robot_manager.avg_yaw) + ' '
                    + str(robot_manager.avg_x) + ' '
                    + str(robot_manager.avg_y) + ' '
                    + str(robot_manager.avg_z) + '\n')
            robot_manager.second += 1
            robot_manager.avg_roll = 0
            robot_manager.avg_pitch = 0
            robot_manager.avg_yaw = 0
            robot_manager.avg_x = 0
            robot_manager.avg_y = 0
            robot_manager.avg_z = 0
            robot_manager.count_group = 1
