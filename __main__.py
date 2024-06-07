# -*- coding: utf-8 -*-

"""

DEEP LEARNING CARS
Simple simulation in Python in which a Neural Network learns to drive a racing car on a track.

Neural network has several inputs (distance sensors and car speed)
and it outputs acceleration and steering.

I used a simple Evolutionary algorithm to train the NN.

- Tomáš Březina 2020

Run command:
pip install -r requirements.txt
"""

# from messages import ask_load_agent, ask_yes_no
from messages import ask_yes_no
from app import App, load_json
from evolution import Evolution, Entity

# Simulation settings
settings = load_json("config.json")
entity = Entity()
SAVE_FILE = False

# Ask the user if they want to load a saved agent
# if ask_yes_no(title="Start", message="Load saved agent?"):
#     SAVE_FILE = ask_load_agent("saves")
#     print("Loading saved agent")
# else:
#     print("Creating a new agent")

# Load the saved agent if a file is specified
if SAVE_FILE:
    entity.load_file(SAVE_FILE)
else:
    nn_stg = load_json("default_agent_config.json")
    if nn_stg:  # Check if nn_stg is not False
        entity.set_parameters_from_dict(nn_stg)
    else:
        print("Creating a new agent")
        # Set default values if the file doesn't exist
        entity.name = "New Agent"
        entity.acceleration = 1
        entity.friction = 0.95
        entity.max_speed = 60
        entity.rotation_speed = 4
        entity.shape = [6, 4, 3, 2]
        entity.max_score = 0
        entity.gen_count = 0

    entity.agent = entity.get_random_agent()

# Create the app and start the simulation
app = App(settings)
app.start_simulation(
    entity=entity,
    track=app.tile_manager.generate_track(shape=(5, 3), spawn_index=0),
    track_filename="track.pkl.npz"
)
