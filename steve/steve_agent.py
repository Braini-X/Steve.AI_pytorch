import math
from past.utils import old_div
import actions
import time
import mob_dict
import json
try:
    from malmo import MalmoPython
except:
    import MalmoPython

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
time_multiplier = int(config.get('DEFAULT', 'TIME_MULTIPLIER'))


class Steve(object):
    def __init__(self, mob_type):
        print("creating new steve.ai placeholder")
        self.mob_type = None
        self.mob_height = None
        self.set_mob_details(mob_type)
        self.target = None
        self.entities = None

    def master_lock(self, ob, agent_host):
        agent_info = (ob.get(u'XPos', 0), ob.get(u'YPos', 0), ob.get(u'ZPos', 0))
        self.get_mob_loc(ob)
        self.closest_enemy(agent_info, self.entities)
        if (self.check_entities() == False):
            return False
        target_yaw, target_pitch = self.calcYawAndPitchToMob(self.entities[self.target], 
            agent_info[0], agent_info[1], agent_info[2], self.mob_height)
        pointing = self.lock_on(agent_host, ob, target_pitch, target_yaw, 3)
        return True


    def lock_on(self, agent_host, ob, target_pitch, target_yaw, threshhold):
        pitch = ob.get(u'Pitch', 0)
        yaw = ob.get(u'Yaw', 0)
        delta_yaw = self.angvel(target_yaw, yaw, 25.0)
        delta_pitch = self.angvel(target_pitch, pitch, 25.0)
        agent_host.sendCommand("turn " + str(delta_yaw/(time_multiplier)*1.5))
        agent_host.sendCommand("pitch " + str(delta_pitch/(time_multiplier)*1.5))
        # if abs(pitch - target_pitch) + abs(yaw - target_yaw) < threshhold:
        #     agent_host.sendCommand("turn 0")
        #     agent_host.sendCommand("pitch 0")
        #     return True
        return False

    def angvel(self, target, current, scale):
        delta = target - current
        while delta < -180:
            delta += 360
        while delta > 180:
            delta -= 360
        return (old_div(2.0, (1.0 + math.exp(old_div(-delta, scale))))) - 1.0

    def calcYawAndPitchToMob(self, target, x, y, z, target_height):
        dx = target[0] - x
        dz = target[2] - z
        yaw = -180 * math.atan2(dx, dz) / math.pi
        distance = math.sqrt(dx * dx + dz * dz)
        pitch = math.atan2(((y + 1.625) - (target[1] + target_height * 1.5)), distance) * 180.0 / math.pi
        return yaw, pitch

    def get_mob_loc(self, ob):
        """gets the locations of all the entities in world state"""
        entities = {}
        for ent in ob["entities"]:
            if (ent["name"] == self.mob_type):
                mob_id = ent['id']
                entities[mob_id] = (ent['x'], ent['y'], ent['z'], ent['life'], ent['motionX'], ent['motionZ'])
        self.entities = entities

    def closest_enemy(self, agent, entities):
        mob_id = ""
        dist = 10000
        for mobs in entities.keys():
            new_dist = self.calculate_distance(agent, entities[mobs])
            if (dist > new_dist):
                mob_id = mobs
                dist = new_dist
        self.target = mob_id

    def calculate_distance(self, agent, mob):
        """Takes the agent and mob's location and calculates distance"""
        return math.sqrt((agent[0] - mob[0]) ** 2 + (agent[2] - mob[2]) ** 2)
    
    def calculate_lava(self, agent):
#         print(agent[0])
        cent_dist = math.sqrt((agent[0] - 0) ** 2 + (agent[2] - 0) ** 2)
        if cent_dist < 6:
            cent_dist = 0
        return cent_dist
    
    def in_lava(self, agent):
        x = agent[0]
        z = agent[2]
        if (x > 10 or x < -10 or z > 10 or z < -10):
            return True
        else:
            return False

    def perform_action(self, agent_host, action):
        action_fraction = .9
        if action == actions.MOVE_LEFT:
            # print("moving left")
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("strafe 0")
            agent_host.sendCommand("strafe -.5")
            time_to_block = (float(config.get('DEFAULT', 'TIME_STEP')) / time_multiplier) * action_fraction
            #time.sleep(time_to_block)
        elif action == actions.MOVE_RIGHT:
            # print("moving right")
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("strafe 0")
            agent_host.sendCommand("strafe .5")
            time_to_block = (float(config.get('DEFAULT', 'TIME_STEP')) / time_multiplier) * action_fraction
            #time.sleep(time_to_block)
        elif action == actions.MOVE_FORWARD:
            # print("moving forward")
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("strafe 0")
            agent_host.sendCommand("move .5")
            time_to_block = (float(config.get('DEFAULT', 'TIME_STEP')) / time_multiplier) * action_fraction
            #time.sleep(time_to_block)
        elif action == actions.MOVE_BACKWARD:
            # print("moving backward")
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("strafe 0")
            agent_host.sendCommand("move -.5")
            time_to_block = (float(config.get('DEFAULT', 'TIME_STEP')) / time_multiplier) * action_fraction
           # time.sleep(time_to_block)
        elif action == actions.STRIKE:
            # print("striking")
            agent_host.sendCommand("attack 1")
            time_to_strike = (float(config.get('DEFAULT', 'TIME_STEP')) / time_multiplier) * action_fraction
            time.sleep(time_to_strike)
            agent_host.sendCommand("attack 0")
        elif action == actions.BLOCK:
            # print("blocking")
            agent_host.sendCommand("use 1")
            time_to_block = 5*(float(config.get('DEFAULT', 'TIME_STEP')) / time_multiplier) * action_fraction
            time.sleep(time_to_block)
            agent_host.sendCommand("use 0")
        elif action == actions.JUMP:
            # print("jumping")
            agent_host.sendCommand("attack 1")
            time_to_block = (float(config.get('DEFAULT', 'TIME_STEP')) / time_multiplier) * action_fraction
            time.sleep(time_to_block)
            agent_host.sendCommand("attack 0")
        else:
            print("INVALID ACTION: " + str(action))

        # return new state, reward, and whether mission is done

    def get_state(self, ob, time_alive):
        ''' 0: Life
            1: Time Alive
            2: Agent X
            3: Agent Z
            4: Horde_health
            5: Target X
            6: Target Z'''
        if (self.check_entities() == False):
            agent_info = (ob.get(u'XPos', 0), ob.get(u'YPos', 0), ob.get(u'ZPos', 0))
            self.get_mob_loc(ob)
            self.closest_enemy(agent_info, self.entities)
        horde = self.horde_health()
        target_x, target_z = self.entities[self.target][0], self.entities[self.target][2]
        return [float(round(ob["Life"])), float(time_alive), float(round(ob["XPos"])),
                float(round(ob["ZPos"])), float(round(horde)), float(round(target_x)), 
                float(round(target_z))]

    def check_entities(self):
        if (len(self.entities.keys()) < 1):
            return False
        elif (self.target != None and self.target not in self.entities.keys()):
            return False
        return True

    def set_mob_details(self, mob_type):
        if (mob_type == 'zombie'):
            self.mob_type = 'Zombie'
            self.mob_height = mob_dict.ZOMBIE
        elif(mob_type == 'spider'):
            self.mob_type = 'Spider'
            self.mob_height = mob_dict.SPIDER 
        elif(mob_type == 'skeleton'):
            self.mob_type = 'Skeleton'
            self.mob_height = mob_dict.SKELETON
        elif(mob_type == 'giant'):
            self.mob_type = 'Giant'
            self.mob_height = mob_dict.GIANT

    def horde_health(self):
        res = 0
        for mob in self.entities.keys():
            res += self.entities[mob][3]
        return res