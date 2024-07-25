import itertools
import numpy
import time
import threading
import pypot.dynamixel

motor_ids = {"tower_1": 1, "tower_2": 2, "tower_3": 3, "base": 4}


ports = pypot.dynamixel.get_available_ports()
print('available ports:', ports)



if not ports:
    raise IOError('No port available.')

port = ports[0]
print('Using the first on the list', port)

dxl_io = pypot.dynamixel.Dxl320IO(port)
print('Connected!')

found_ids = dxl_io.scan()
print('Found ids:', found_ids)

if len(found_ids) < 2:
    raise IOError('You should connect at least two motors on the bus for this test.')

ids = found_ids

dxl_io.enable_torque(ids)

speed = dict(zip(ids, itertools.repeat(200)))
dxl_io.set_moving_speed(speed)
motor_positions = dict(zip(ids, dxl_io.get_present_position(ids)))
print("motor positions: ", motor_positions)

def motor_goto_duration(motor_name, goal_position, duration = 1.0):
    motor_id = motor_ids[motor_name]
    # start_position = dxl_io.get_present_position([motor_id])[0]
    start_position = motor_positions[motor_id]
    delta_position = goal_position - start_position
    num_steps = duration/0.05
    step_size = delta_position/num_steps
    print(num_steps)
    for t in range(int(num_steps)):
        dxl_io.set_goal_position({motor_id: start_position + step_size*(t+1)})
        time.sleep(0.05)
    motor_positions[motor_id] = goal_position

def multi_motor_goto_duration(motor_list, goal_positions, duration = 1.0):
    num_steps = duration/0.05
    num_motors = len(motor_list)

    active_motor_ids = []
    start_positions = []
    delta_positions = []
    step_sizes = []

    for i in range(num_motors):
        motor_id = motor_ids[motor_list[i]]
        active_motor_ids.append(motor_id)

        start_position = motor_positions[motor_id]
        start_positions.append(start_position)

        delta_positions.append(goal_positions[i] - start_position)
        step_sizes.append(delta_positions[i]/num_steps)

    print(num_steps)

    for t in range(int(num_steps)):
        # for i in range(num_motors):
        #     dxl_io.set_goal_position({active_motor_ids[i]: start_positions[i] + step_sizes[i]*(t+1)})
        intermediate_positions = []
        for i in range(num_motors):
            intermediate_positions.append(start_positions[i] + step_sizes[i]*(t+1))
        dxl_io.set_goal_position(dict(zip(active_motor_ids, intermediate_positions)))
        time.sleep(0.05)

    for i in range(num_motors):
        motor_positions[active_motor_ids[i]] = goal_positions[i]

def motor_goto(motor_name, goal_position):
    motor_id = motor_ids[motor_name]
    dxl_io.set_goal_position({motor_id: goal_position})
    motor_positions[motor_id] = goal_position
    time.sleep(0.1)


def reset():
    multi_motor_goto_duration(["base", "tower_1", "tower_2", "tower_3"], [0, 40, 50, 50], 1.0)

# bl = Blossom()
# bl.connect() # safe init and connects to blossom and puts blossom in reset position
def re_engage():
    motor_list = ["tower_1", "tower_2", "tower_3"]

    reset()

    motor_goto_duration("base", 20, 0.5)
    # # motor_goto("tower_3", -10)
    # # motor_goto("tower_1", 10)
    # # motor_goto("tower_2", 30)
    multi_motor_goto_duration(motor_list, [-40, 30, -10])

    time.sleep(2)

    motor_goto_duration("base", -5, 0.5)
    # # motor_goto_duration("tower_1", 100, 1.0)
    # # motor_goto_duration("tower_3", -40, 0.75)
    # # motor_goto_duration("tower_2", -40, 0.75)
    multi_motor_goto_duration(motor_list, [90, -50, -50], 1)

    time.sleep(3)

    multi_motor_goto_duration(motor_list, [20, 40, 40], 0.5)
    time.sleep(0.75)
    motor_goto_duration("base", -70, 0.2)
    # # motor_goto_duration("tower_1", 80, 1.5)
    # # motor_goto_duration("tower_3", 20, 0.5)
    # # motor_goto_duration("tower_2", 20, 0.5)

    time.sleep(1)

    motor_goto_duration("base", 0, 0.2)

    time.sleep(1.25)

    motor_goto_duration("base", -70, 0.2)

    time.sleep(1.5)

    multi_motor_goto_duration(["base", "tower_2", "tower_1", "tower_3"], [0, -75, 80, 70], 1.7)
    # # motor_goto_duration("tower_2", -40, 1.0)
    # # motor_goto_duration("tower_3", 70, 0.5)
    # # motor_goto("tower_1", 80)


    time.sleep(3)

    reset()
    time.sleep(0.5)
    reset()

    time.sleep(2)