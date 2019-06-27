"""
Revolve body generator based on RoboGen framework
"""
import yaml
import traceback
from collections import OrderedDict

from pyrevolve import SDF

from .revolve_module import CoreModule, TouchSensorModule, Orientation
from .revolve_module import Orientation
from .brain import Brain, BrainNN

from .render.render import Render
from .render.brain_graph import BrainGraph
from .measure.measure_body import MeasureBody
from .measure.measure_brain import MeasureBrain

from ..custom_logging.logger import logger


class RevolveBot:
    """
    Basic robot description class that contains robot's body and/or brain
    structures, ID and several other necessary parameters. Capable of reading
    a robot's sdf mode
    """

    def __init__(self, _id=None):
        self._id = _id
        self._body = None
        self._brain = None
        self._morphological_measurements = None
        self._brain_measurements = None
        self._behavioural_measurements = None
        # self._battery_level = None

    @property
    def id(self):
        return self._id

    @property
    def body(self):
        return self._body

    @property
    def brain(self):
        return self._brain

    def size(self):
        robot_size = 1 + self._recursive_size_measurement(self._body)
        return robot_size

    def _recursive_size_measurement(self, module):
        count = 0
        for _, child in module.iter_children():
            if child is not None:
                count += 1 + self._recursive_size_measurement(child)

        return count

    def measure_behaviour(self):
        """

        :return:
        """
        pass

    def measure_phenotype(self, experiment_name):
        self._morphological_measurements = self.measure_body()
        self._brain_measurements = self.measure_brain()
        logger.info('Robot ' + str(self.id) + ' was measured.')
        self.export_phenotype_measurements(experiment_name)

    def measure_body(self):
        """
        :return: instance of MeasureBody after performing all measurements
        """
        if self._body is None:
            raise RuntimeError('Body not initialized')
        try:
            measure = MeasureBody(self._body)
            measure.measure_all()
            return measure
        except Exception as e:
            logger.exception('Failed measuring body')

    def export_phenotype_measurements(self, path):
        with open('experiments/' + path + '/data_fullevolution/descriptors/'
                  + 'phenotype_desc_' + str(self.id) + '.txt', 'w+') as file:
            # TODO this crashes
            for key, value in self._morphological_measurements.measurements_to_dict().items():
                file.write('{} {}\n'.format(key, value))
            for key, value in self._brain_measurements.measurements_to_dict().items():
                file.write('{} {}\n'.format(key, value))

    def measure_brain(self):
        """
        :return: instance of MeasureBrain after performing all measurements
        """
        try:
            measure = MeasureBrain(self._brain, 10)
            measure_b = MeasureBody(self._body)
            measure_b.count_active_hinges()
            if measure_b.active_hinges_count > 0:
                measure.measure_all()
            else:
                measure.set_all_zero()
            return measure
        except Exception as e:
            logger.exception('Failed measuring brain')


    def load(self, text, conf_type):
        """
        Load robot's description from a string and parse it to Python structure
        :param text: Robot's description string
        :param conf_type: Type of a robot's description format
        :return:
        """
        if 'yaml' == conf_type:
            self.load_yaml(text)
        elif 'sdf' == conf_type:
            raise NotImplementedError("Loading from SDF not yet implemented")

    def load_yaml(self, text):
        """
        Load robot's description from a yaml string
        :param text: Robot's yaml description
        """
        yaml_bot = yaml.safe_load(text)
        self._id = yaml_bot['id'] if 'id' in yaml_bot else None
        self._body = CoreModule.FromYaml(yaml_bot['body'])

        try:
            if 'brain' in yaml_bot:
                yaml_brain = yaml_bot['brain']
                if 'type' not in yaml_brain:
                    # raise IOError("brain type not defined, please fix it")
                    yaml_brain['type'] = 'neural-network'
                self._brain = Brain.from_yaml(yaml_brain)
            else:
                self._brain = Brain()
        except:
            self._brain = Brain()
            logger.exception('Failed to load brain, setting to None')

    def load_file(self, path, conf_type='yaml'):
        """
        Read robot's description from a file and parse it to Python structure
        :param path: Robot's description file path
        :param conf_type: Type of a robot's description format
        :return:
        """
        with open(path, 'r') as robot_file:
            robot = robot_file.read()

        self.load(robot, conf_type)

    def to_sdf(self, pose=SDF.math.Vector3(0, 0, 0.25), nice_format=None):
        if type(nice_format) is bool:
            nice_format = '\t' if nice_format else None
        return SDF.revolve_bot_to_sdf(self, pose, nice_format)

    def to_yaml(self):
        """
        Converts robot data structure to yaml

        :return:
        """
        yaml_dict = OrderedDict()
        yaml_dict['id'] = self._id
        yaml_dict['body'] = self._body.to_yaml()
        if self._brain is not None:
            yaml_dict['brain'] = self._brain.to_yaml()

        return yaml.dump(yaml_dict)

    def save_file(self, path, conf_type='yaml'):
        """
        Save robot's description on a given file path in a specified format
        :param path:
        :param conf_type:
        :return:
        """
        robot = ''
        if 'yaml' == conf_type:
            robot = self.to_yaml()
        elif 'sdf' == conf_type:
            robot = self.to_sdf(nice_format=True)

        with open(path, 'w') as robot_file:
            robot_file.write(robot)

    def update_substrate(self, raise_for_intersections=False):
        """
        Update all coordinates for body components

        :param raise_for_intersections: enable raising an exception if a collision of coordinates is detected
        :raises self.ItersectionCollisionException: If a collision of coordinates is detected (and check is enabled)
        """
        substrate_coordinates_map = {(0, 0): self._body.id}
        self._body.substrate_coordinates = (0, 0)
        self._update_substrate(raise_for_intersections, self._body, Orientation.NORTH, substrate_coordinates_map)

    class ItersectionCollisionException(Exception):
        """
        A collision has been detected when updating the robot coordinates.
        Check self.substrate_coordinates_map to know more.
        """
        def __init__(self, substrate_coordinates_map):
            super().__init__(self)
            self.substrate_coordinates_map = substrate_coordinates_map

    def _update_substrate(self,
                          raise_for_intersections,
                          parent,
                          parent_direction,
                          substrate_coordinates_map):
        """
        Internal recursive function for self.update_substrate()
        :param raise_for_intersections: same as in self.update_substrate
        :param parent: updates the children of this parent
        :param parent_direction: the "absolute" orientation of this parent
        :param substrate_coordinates_map: map for all already explored coordinates(useful for coordinates conflict checks)
        """
        dic = {Orientation.NORTH: 0,
               Orientation.WEST:  1,
               Orientation.SOUTH: 2,
               Orientation.EAST:  3}
        inverse_dic = {0: Orientation.NORTH,
                       1: Orientation.WEST,
                       2: Orientation.SOUTH,
                       3: Orientation.EAST}

        movement_table = {
            Orientation.NORTH: ( 1,  0),
            Orientation.WEST:  ( 0, -1),
            Orientation.SOUTH: (-1,  0),
            Orientation.EAST:  ( 0,  1),
        }

        for slot, module in parent.iter_children():
            if module is None:
                continue

            slot = Orientation(slot)

            # calculate new direction
            direction = dic[parent_direction] + dic[slot]
            if direction >= len(dic):
                direction = direction - len(dic)
            new_direction = Orientation(inverse_dic[direction])

            # calculate new coordinate
            movement = movement_table[new_direction]
            coordinates = (
                parent.substrate_coordinates[0] + movement[0],
                parent.substrate_coordinates[1] + movement[1],
            )
            module.substrate_coordinates = coordinates

            # For Karine: If you need to validate old robots, remember to add this condition to this if:
            # if raise_for_intersections and coordinates in substrate_coordinates_map and type(module) is not TouchSensorModule:
            if raise_for_intersections:
                if coordinates in substrate_coordinates_map:
                    raise self.ItersectionCollisionException(substrate_coordinates_map)
                substrate_coordinates_map[coordinates] = module.id

            self._update_substrate(raise_for_intersections,
                                   module,
                                   new_direction,
                                   substrate_coordinates_map)

    def render_brain(self, img_path):
        """
        Render image of brain
        @param img_path: path to where to store image
        """
        if self._brain is None:
            raise RuntimeError('Brain not initialized')
        elif isinstance(self._brain, BrainNN):
            try:
                brain_graph = BrainGraph(self._brain, img_path)
                brain_graph.brain_to_graph(True)
                brain_graph.save_graph()
            except Exception as e:
                logger.exception('Failed rendering brain. Exception:')
        else:
            raise RuntimeError('Brain {} image rendering not supported'.format(type(self._brain)))

    def render_body(self, img_path):
        """
        Render 2d representation of robot and store as png
        :param img_path: path of storing png file
        """
        if self._body is None:
            raise RuntimeError('Body not initialized')
        else:
            try:
                render = Render()
                render.render_robot(self._body, img_path)
            except Exception as e:
                logger.exception('Failed rendering 2d robot')

    def __repr__(self):
        return f'RevolveBot({self.id})'
