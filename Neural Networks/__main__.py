from messages import ask_load_nn, ask_yes_no
from app import App, load_json
from evolution import Evolution, Entity

# simulation settings
settings = load_json("config.json")
entity = Entity()

#SAVE_FILE = False
#
#if ask_yes_no(title="Start",message="Load saved NN?"):
 #   SAVE_FILE = ask_load_nn("saves")

# if save file is  defined
SAVE_FILE = "./saves/x1.json"
if SAVE_FILE:
    entity.load_file(SAVE_FILE)
else:
    # create new neural network
    nn_stg = load_json("default_nn_config.json")
    entity.set_parameters_from_dict(nn_stg)
    entity.nn = entity.get_random_nn();

# window
app = App(settings)
app.start_simulation(
    entity=entity,
    track=app.tile_manager.generate_track(shape=(5,3), spawn_index=0)
)
