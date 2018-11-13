"""
    This is the main file that starts the simulation.
    It contains a "World" object specifying the world settings, and a set of particles.
    Every time the robot moves, it generates random observations usded to update the particle sets
    to estimate the robot path as well as the landmarks locations.
"""
import sys
import random
import math
from copy import deepcopy
from world import World
from world import WINDOWHEIGHT, WINDOWWIDTH
from particle import Particle
import slam_helper
import matplotlib.pyplot as plt

class FastSlam(object):
    """Main class that implements the FastSLAM1.0 algorithm"""
    def __init__(self, x, y, orien, particle_size):
        self.world = World()
        self.particles = [Particle(x, y, random.random()* 2.*math.pi) for i in range(particle_size)]
        self.robot = Particle(x, y, orien, is_robot=True)
        self.particle_size = particle_size
        self.c = 0
        self.error = []
        self.lm_error = []

    def run_simulation(self):
        while True:
            for event in self.world.pygame.event.get():
                self.world.test_end(event)
            self.world.clear()
            key_pressed = self.world.pygame.key.get_pressed()

            if self.world.move_forward(key_pressed):
                self.c += 1
                if self.c < 50:
                    self.move_forward(2)
                    obs = self.robot.sense(self.world.landmarks, 2)
                    #print('obs: ', obs)
                    avg_err = 0
                    for p in self.particles:
                        p.update(obs)
                        avg_err += p.lm_error
                    avg_err /= self.particle_size
                    self.lm_error.append(avg_err)

                    self.particles = self.resample_particles()
                    landmarks = self.get_predicted_landmarks()
                    #for i in range(len(landmarks)):
                        #print('landmark: ', landmarks[i])
                else:
                    t1 = list(range(self.c - 1))
                    plt.figure(0)
                    plt.plot(t1, self.error)
                    plt.xlabel('time')
                    plt.ylabel('Localization Error')
                    plt.legend(
                        ('Particles = 5', 'Particles = 10', 'Particles = 20', 'Particles = 100', 'Particles = 200'),
                        loc='upper right', shadow=True)
                    plt.figure(1)
                    plt.plot(t1, self.lm_error)
                    plt.xlabel('time')
                    plt.ylabel('Mapping Error')
                    plt.legend(
                        ('Particles = 5', 'Particles = 10', 'Particles = 20', 'Particles = 100', 'Particles = 200'),
                        loc='upper right', shadow=True)
                    break
            if self.world.turn_left(key_pressed):
                self.turn_left(5)
            if self.world.turn_right(key_pressed):
                self.turn_right(5)
            self.world.render(self.robot, self.particles, self.get_predicted_landmarks())

    def move_forward(self, step):
        self.robot.forward(step)
        for p in self.particles:
            p.forward(step)

    def turn_left(self, angle):
        self.robot.turn_left(angle)
        for p in self.particles:
            p.turn_left(angle)

    def turn_right(self, angle):
        self.robot.turn_right(angle)
        for p in self.particles:
            p.turn_right(angle)

    def resample_particles(self):
        new_particles = []
        weight = [p.weight for p in self.particles]
        index = int(random.random() * self.particle_size)
        beta = 0.0
        mw = max(weight)
        for i in range(self.particle_size):
            beta += random.random() * 2.0 * mw
            while beta > weight[index]:
                beta -= weight[index]
                index = (index + 1) % self.particle_size
            new_particle = deepcopy(self.particles[index])
            new_particle.weight = 1
            new_particles.append(new_particle)
        #print('robot: ', [self.robot.pos_x, self.robot.pos_y])
        #print('particle: ', [new_particle.pos_x, new_particle.pos_y])
        self.error.append(slam_helper.euclidean_distance([self.robot.pos_x, self.robot.pos_y], [new_particle.pos_x, new_particle.pos_y]))
        return new_particles

    def get_predicted_landmarks(self):
        return self.particles[0].landmarks

    def check_envr_bounds(self, x, y):
        out_flag = False
        if x <= WINDOWWIDTH and y <= WINDOWHEIGHT:
            out_flag = True
        return out_flag


if __name__=="__main__":
    loc_err_part = []
    map_err_part = []
    t1 = [5, 10, 20, 100, 200]
    for i in t1:
        random.seed(5)
        simulator = FastSlam(80, 140, 0, particle_size=i)
        simulator.run_simulation()
        loc_err_part.append(sum(simulator.lm_error) / len(simulator.lm_error))
        map_err_part.append(sum(simulator.error) / len(simulator.error))
    print('Average Localization error for 5,10, 20, 100, 200: ', loc_err_part)
    print('Average Mapping error for 5,10, 20, 100, 200: ', map_err_part)

    plt.figure(2)
    plt.plot(t1, map_err_part, '-o')
    plt.xticks(t1, [t for t in t1])
    #plt.title('Average Mapping Error vs Number of Particles')
    plt.plot(t1, loc_err_part, '-o')
    plt.xticks(t1, [t for t in t1])
    plt.xlabel('Number of Particles')
    plt.ylabel('Average Error')
    plt.title('Average Error vs Number of Particles')
    plt.legend(
        ('Average Mapping Error', 'Average Localization Error',),
        loc='upper right', shadow=True)
    plt.show()
