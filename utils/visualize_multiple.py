from src.EA.CMAES import CMAES, CMAES_opts
from src.EA.NSGA import NSGAII, NSGA_opts
from src.world.World import World
from src.world.robot.controllers import MLP
from src.world.robot.morphology.AntCustomRobot import AntRobot
from src.utils.Filesys import get_project_root
from gymnasium.vector import AsyncVectorEnv

import xml.etree.ElementTree as xml
import gymnasium as gym
import numpy as np
import os

from time import sleep

""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.

    Exercise3 body-brain: This is your first full body+brain evolution by adapting a custom Ant-v5 gym environment. 
    We adjust both leg lengths and controller weights for a locomotion task.   
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'

ADDED_LEGS = 2 # per side
BODY_PARAMS = 25 # 8 leg switches, 2*8 parameters per leg, half sausage lenght


class AntWorld(World):
    def __init__(self, ):
        action_space = 8+2*2*ADDED_LEGS  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 27+ADDED_LEGS*4+ADDED_LEGS*4  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space


        self.n_repeats = 5
        self.n_steps = 1000
        self.controller = MLP.NNController(state_space, action_space)
        #print(f"Controller weights: {self.controller.n_params}")
        #print(f"Genome size: {self.controller.n_params + BODY_PARAMS}")
        self.n_weights = self.controller.n_params

        self.leg_switches = np.zeros(10)

        self.n_params = self.n_weights + BODY_PARAMS
        self.world_file = os.path.join(ROOT_DIR, "AntEnv.xml")

        self.joint_limits = [[-30, 30], [30, 70],
                             [-30, 30], [30, 70],
                             [-30, 30], [-70, -30],
                             [-30, 30], [-70, -30],
                             [-30, 30], [-70, -30],
                             [-30, 30], [-70, -30],
                             [-30, 30], [30, 70],
                             [-30, 30], [30, 70],
                             ]
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           [0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           [0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           [0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           ]

    def geno2pheno(self, genotype):
        control_weights = genotype[:self.n_weights]
        body_params = genotype[self.n_weights:]
        assert len(body_params) == BODY_PARAMS
        assert len(control_weights) == self.n_weights

        self.controller.geno2pheno(control_weights)

        num_legs = int((BODY_PARAMS-1)/3)
        # print(f"Body params: {body_params}")
        self.leg_switches = (body_params[:num_legs] + 1.0) / 2
        # print(f"Leg switches: {self.leg_switches}")
        self.leg_params = (body_params[num_legs:-1] + 1.5) / 5 * 0.5 + 0.1 # ordered as [leg_1_thigh_left, leg_1_ankle_left, leg_1_thigh_right, leg_1_ankle_right, leg_2_thigh_left, leg_2_ankle_left, leg_2_thigh_right, leg_2_ankle_right, ...]
        assert not np.any(self.leg_params <= 0)
        half_sausage_length = (body_params[-1] + 1) / 2.02 + 0.01
        #print(f"Half sausage length: {half_sausage_length}")
        self.leg_params = np.reshape(self.leg_params, (num_legs, 2)) # 8 legs, 2 parameters each (thigh and ankle)
        self.leg_param = np.reshape(self.leg_params, (int(num_legs/2), 2, 2)) # 4 leg positions along the body, 1 per side, 2 parameters each (thigh and ankle)

        # Define the 3D coordinates of the relative tree structure
        legs_per_side = int(num_legs/2)
        left_hip_xyz = np.zeros((legs_per_side,3))
        left_knee_xyz = np.zeros((legs_per_side,3))
        left_toe_xyz = np.zeros((legs_per_side,3))
        right_hip_xyz = np.zeros((legs_per_side,3))
        right_knee_xyz = np.zeros((legs_per_side,3))
        right_toe_xyz = np.zeros((legs_per_side,3))

        connectivity_mat = np.zeros((num_legs*3, num_legs*3))

        body_length = half_sausage_length * 2 # capsule length * 2 to ensure even split of the legs
        body_radius = 0.25 # capsule radius so that the legs do not start inside the body

        legs_per_side = int(num_legs / 2)
        margin = body_length / (legs_per_side * 2 + 2)
        x_offsets = np.linspace(-body_length / 2 + margin, body_length / 2 - margin, legs_per_side)

        for i in range(legs_per_side):
            x_pos = x_offsets[i]
            thigh_l, ankle_l = self.leg_param[i][0]
            thigh_r, ankle_r = self.leg_param[i][1]

            # left legs
            if self.leg_switches[2 * i] >= 0.5: # if leg there (switch on)
                left_hip_xyz[i] = np.array([x_pos,  body_radius, 0])

                left_knee_xyz[i] = left_hip_xyz[i] + np.array([0, thigh_l, -thigh_l * 0.5]) # thigh outward in +y, pointing a bit down in -z
                left_toe_xyz[i] = left_knee_xyz[i] + np.array([0, 0, -ankle_l]) # ankle straight down in -z

                connectivity_mat[6*i, 6*i] = 150
                connectivity_mat[6*i+1, 6*i+1] = 150
                connectivity_mat[6*i, 6*i+1] = np.inf
                connectivity_mat[6*i+1, 6*i+2] = np.inf
            else: # if leg not there (switch off)
                left_hip_xyz[i] = left_knee_xyz[i] = left_toe_xyz[i] = np.zeros(3)

            # right legs
            if self.leg_switches[2 * i + 1] >= 0.5: # if leg there (switch on)
                right_hip_xyz[i] = np.array([x_pos, -body_radius, 0])

                right_knee_xyz[i] = right_hip_xyz[i] + np.array([0, -thigh_r, -thigh_r * 0.5]) # thigh outward in -y, pointing a bit down in -z
                right_toe_xyz[i] = right_knee_xyz[i] + np.array([0, 0, -ankle_r]) # ankle straight down in -z

                connectivity_mat[6*i+3, 6*i+3] = 150
                connectivity_mat[6*i+4, 6*i+4] = 150
                connectivity_mat[6*i+3, 6*i+4] = np.inf
                connectivity_mat[6*i+4, 6*i+5] = np.inf
            else: # if leg not there (switch off)
                right_hip_xyz[i] = right_knee_xyz[i] = right_toe_xyz[i] = np.zeros(3)

            points = []
            for i in range(4):
                points.append(left_hip_xyz[i])
                points.append(left_knee_xyz[i])
                points.append(left_toe_xyz[i])
                points.append(right_hip_xyz[i])
                points.append(right_knee_xyz[i])
                points.append(right_toe_xyz[i])
        
        points = np.vstack(points)
        #print(f"Points: {points}")
        #print(f"Connectivity matrix: {connectivity_mat}")
        return points, connectivity_mat, half_sausage_length

    def evaluate_individual(self, genotype):
        points, connectivity_mat, half_sausage_length = self.geno2pheno(genotype)      

        robot = AntRobot(points, connectivity_mat, half_sausage_length, self.joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml()

        # % Defining the Robot environment in MuJoCo
        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding='unicode')

        with open(self.world_file, "w") as f:
            f.write(world_xml)

        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=0.1,
                    max_episode_steps=self.n_steps,
                )
                for i_env in range(self.n_repeats)
            ]
        )

        rewards_full = np.zeros((self.n_steps, self.n_repeats))
        multi_obj_rewards_full = np.zeros((self.n_steps, self.n_repeats, 2))

        observations, info = envs.reset()
        reshaped_obs = _reshape_observations(observations, self.leg_switches)
        done_mask = np.zeros(self.n_repeats, dtype=bool)
        for step in range(self.n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(reshaped_obs.T).T)
            reshaped_action = _reshape_actions(actions, self.leg_switches)
            #print(f"Original action: {actions}")
            #print(f"Reshaped action: {reshaped_action}")
            observations, rewards, dones, truncated, infos = envs.step(reshaped_action)
            reshaped_obs = _reshape_observations(observations, self.leg_switches)

            # Store rewards for active environments only
            rewards_full[step, done_mask == False] = rewards[done_mask == False]

            x_velocities = np.array(infos['reward_forward'])
            distance_reward = np.array(infos['reward_distance'])
            # Control cost is usually negative or zero, we want to maximize -cost
            control_costs = np.array(infos['ctrl_cost'])
            total_volume = np.array(infos['total_volume'])
            #multi_obj_reward_step = np.array([x_velocities, -total_volume]).T
            x_velocities_signed_pow = np.sign(x_velocities) * np.abs(x_velocities)**2
            multi_obj_reward_step = np.array([distance_reward, -control_costs/10]).T
            multi_obj_rewards_full[step, done_mask == False] = multi_obj_reward_step[done_mask == False]

            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


def run_EA_single(ea_single, world):
    for gen in range(ea_single.n_gen):
        print(f"Generation {gen}")
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        for index, genotype in enumerate(pop):
            print(f"Evaluating individual {index}")
            fit_ind, _ = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_single.tell(pop, fitnesses_gen)


def run_EA_multi(ea_multi, world):
    for gen in range(ea_multi.n_gen):
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        print(f"Generation {gen}")
        for index, genotype in enumerate(pop):
            print(f"Evaluating individual {index}")
            _, fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea_multi.tell(pop, fitnesses_gen)


def generate_best_individual_video(world, video_name: str = 'EvoRob3_video.mp4'):
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")
    rewards_list = []

    observations, info = env.reset()
    reshaped_obs = _reshape_observation(observations, world.leg_switches)
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = world.controller.get_action(reshaped_obs)
        reshaped_action = _reshape_action(action, world.leg_switches)
        observations, rewards, terminated, truncated, info = env.step(reshaped_action)
        if np.isnan(observations).any():
            print("NaN in observation!")
        if terminated:
            print("Terminated:", terminated, "Info:", info)
        reshaped_obs = _reshape_observation(observations, world.leg_switches)
        rewards_list.append(rewards)
        if terminated:
            break
    #print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()


def visualise_individual(genotype):
    world = AntWorld()
    points, connectivity_mat, half_sausage_length = world.geno2pheno(genotype)

    robot = AntRobot(points, connectivity_mat, half_sausage_length, world.joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="human")
    rewards_list = []

    observations, info = env.reset()
    #print(observations)
    #print(len(observations))
    reshaped_obs = _reshape_observation(observations, world.leg_switches)
    #print(reshaped_obs)
    #print(len(reshaped_obs))
    #print(reshaped_obs.shape)
    # for _ in range(100):
       # sleep(0.1)
    for step in range(200):
        action = world.controller.get_action(reshaped_obs)
        reshaped_action = _reshape_action(action, world.leg_switches)
        observations, rewards, terminated, truncated, info = env.step(reshaped_action)
        reshaped_obs = _reshape_observation(observations, world.leg_switches)
        rewards_list.append(rewards)
        if terminated:
            break
    env.close()
    #print(np.sum(rewards_list))

def _reshape_action(actions, leg_switches):
    # Remove all the actions for the legs that are not active
    reshaped_act = []
    for i in range(8):
        if leg_switches[i] >= 0.5:
            reshaped_act.extend(actions[2*i:2*i+2])
    return np.array(reshaped_act)

def _reshape_observation(observations, leg_switches):
    # Reshape the observations to match the expected input shape of the controller
    reshaped_obs = np.zeros((27+ADDED_LEGS*4+ADDED_LEGS*4,))
    reshaped_obs[:5] = observations[:5]

    last_idx = 5

    for i in range(8):
        if leg_switches[i] >= 0.5:
            reshaped_obs[5 + i * 2:5 + i * 2 + 2] = observations[last_idx:last_idx + 2]
            last_idx += 2
        else:
            reshaped_obs[5 + i * 2:5 + i * 2 + 2] = 0.0

    idx = 5 + 2 * 8
    reshaped_obs[idx:idx+6] = observations[last_idx:last_idx + 6]
    idx += 6
    last_idx += 6

    for i in range(8):
        if leg_switches[i] >= 0.5:
            reshaped_obs[idx + i * 2:idx + i * 2 + 2] = observations[last_idx:last_idx + 2]
            last_idx += 2
        else:
            reshaped_obs[idx + i * 2:idx + i * 2 + 2] = 0.0

    return reshaped_obs

def _reshape_actions(actions, leg_switches):
    # Remove all the actions for the legs that are not active
    if actions.ndim == 1:
        actions = actions[None, :]
    n_envs = actions.shape[0]
    reshaped = []
    for env_idx in range(n_envs):
        act = actions[env_idx]
        #print(f"Original action: {act}")
        reshaped_act = []
        for i in range(8):
            if leg_switches[i] >= 0.5:
                reshaped_act.extend(act[2*i:2*i+2])
        #print(f"Reshaped action: {reshaped_act}")
        reshaped.append(reshaped_act)
    return np.array(reshaped)

def _reshape_observations(observations, leg_switches):
    # Reshape the observations to match the expected input shape of the controller
    if observations.ndim == 1:
        observations = observations[None, :]
    n_envs = observations.shape[0]
    reshaped = np.zeros((n_envs, 27+ADDED_LEGS*4+ADDED_LEGS*4))
    for env_idx in range(n_envs):
        obs = observations[env_idx]
        reshaped_obs = np.zeros((27+ADDED_LEGS*4+ADDED_LEGS*4,))
        reshaped_obs[:5] = obs[:5]

        last_idx = 5

        for i in range(8):
            if leg_switches[i] >= 0.5:
                reshaped_obs[5 + i * 2:5 + i * 2 + 2] = obs[last_idx:last_idx + 2]
                last_idx += 2
            else:
                reshaped_obs[5 + i * 2:5 + i * 2 + 2] = 0.0

        idx = 5 + 2 * 8
        reshaped_obs[idx:idx+6] = obs[last_idx:last_idx + 6]
        idx += 6
        last_idx += 6

        for i in range(8):
            if leg_switches[i] >= 0.5:
                reshaped_obs[idx + i * 2:idx + i * 2 + 2] = obs[last_idx:last_idx + 2]
                last_idx += 2
            else:
                reshaped_obs[idx + i * 2:idx + i * 2 + 2] = 0.0
        
        reshaped[env_idx] = reshaped_obs

    return reshaped

def main():
    world = AntWorld()

    population_size = 12
    NSGA_opts["min"] = -1
    NSGA_opts["max"] = 1
    NSGA_opts["num_parents"] = population_size
    NSGA_opts["num_generations"] = 2
    NSGA_opts["mutation_prob"] = 0.05
    NSGA_opts["crossover_prob"] = 0.07

    population_size = 40
    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_parents"] = 100
    CMAES_opts["num_generations"] = 100
    CMAES_opts["mutation_sigma"] = 0.1

    # results_dir = os.path.join(ROOT_DIR, 'saves_03-06')
    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single')

    gen = 99

    # %% visualise
    # TODO: Make a video of the best individual, and plot the fitness curve.
    all_individuals = np.load(os.path.join(results_dir, "full_x.npy")).copy()
    all_fitnesses = np.load(os.path.join(results_dir, "full_f.npy")).copy()
    last_gen = all_individuals[gen][np.argsort(all_fitnesses[gen])]
    for ind in last_gen[5:]:
        visualise_individual(ind)

    return
    points, connectivity_mat, half_sausage_length = world.geno2pheno(best_individual)
    robot = AntRobot(points, connectivity_mat, half_sausage_length, world.joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    generate_best_individual_video(world)

    best_10 = np.load(os.path.join(results_dir, f"{NSGA_opts['num_generations']-1}", "x_best_10.npy"))

    for i, individual in enumerate(best_10):
        points, connectivity_mat, half_sausage_length = world.geno2pheno(individual)
        robot = AntRobot(points, connectivity_mat, half_sausage_length, world.joint_limits, world.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml()
            # % Defining the Robot environment in MuJoCo
        world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
        robot_env = world_xml.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding='unicode')
        with open(world.world_file, "w") as f:
            f.write(world_xml)

        generate_best_individual_video(world, video_name=f'EvoRob3_video_{i}.mp4')


if __name__ == '__main__':
    main()