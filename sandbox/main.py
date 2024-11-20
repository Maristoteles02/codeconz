#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd

import engine.engine as engine
import train as train
import time
from bots.ppo_2 import PPO
from bots.randbot import RandBot
from bots.randbot import RandBot
import engine.engine as engine
import train

def generate_randbot_data(num_episodes=10, num_steps=50, output_file="randbot_data.json"):
    """Genera datos usando RandBot y los guarda en un archivo JSON."""

    map_dir = "./maps"
    map_files = sorted([f for f in os.listdir(map_dir) if f.startswith("mapa_") and f.endswith(".txt")])
    cfg_files_train = [os.path.join(map_dir, f) for f in map_files[:18]]

    if not cfg_files_train:
        print("No se encontraron mapas para entrenamiento.")
        return

    data = []
    bots = [RandBot() for _ in range(6)]
    # Inicializar bots antes de comenzar el juego

    for episode in range(num_episodes):
        print(f"\nGenerando episodio {episode + 1}/{num_episodes}")

        for j in range(len(cfg_files_train)):
            config = engine.GameConfig(cfg_files_train[j])
            game = engine.Game(config, len(bots))
            for idx, bot in enumerate(bots):
                bot.initialize({
                    "player_num": idx,
                    "position": game.players[idx].pos,
                    "map": game.island.map,
                    "lighthouses": [lh.pos for lh in game.lighthouses.values()]
                })

            iface = train.Interface([game], bots, debug=False)

            for idx, player in enumerate(game.players):
                bots[idx].player_num = player.num

            for step in range(num_steps):
                states = [iface.get_state(player, idx) for idx, player in enumerate(game.players)]
                actions = bots[0].play(states)

                for idx, bot in enumerate(bots):
                    status = iface.turn(game.players[idx], actions[idx])
                    if not status["success"]:
                        break

    # Guardar datos al finalizar
    bots[0].save_data(output_file)

if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Evita la creación de la ventana gráfica
    #generate_randbot_data(num_episodes=10, num_steps=50)
    # Directorio donde se encuentran los mapas generados
    map_dir = "./maps"

    # Listar todos los archivos en el directorio que cumplan con el formato esperado
    map_files = sorted([f for f in os.listdir(map_dir) if f.startswith("mapa_") and f.endswith(".txt")])

    # Generar nombres simulados de mapas si no se leen directamente del directorio
    map_files = [f"mapa_{i}.txt" for i in range(40)]

    # Seleccionar mapas específicos para evaluación
    evaluation_maps = [18, 19, 38, 39]
    eval_files = [map_files[i] for i in evaluation_maps]

    # Filtrar mapas restantes para entrenamiento
    remaining_maps = [map for i, map in enumerate(map_files) if i not in evaluation_maps]

    # Organizar los mapas restantes en bloques de 20 (18 para entrenamiento, 2 para evaluación)
    train_eval_blocks = []
    for i in range(1, len(remaining_maps), 20):
        train_files = remaining_maps[i:i+18]  # Selecciona 18 mapas para entrenamiento
        block_eval_files = remaining_maps[i+18:i+20]  # Selecciona 2 mapas para evaluación
        train_eval_blocks.append((train_files, block_eval_files))

    # Inicializar listas para los archivos de configuración de entrenamiento y evaluación
    cfg_files_train = []
    cfg_files_eval = []

    # Agregar los mapas de entrenamiento y evaluación con la ruta completa
    for train_files, block_eval_files in train_eval_blocks:
        cfg_files_train.extend([os.path.join(map_dir, f) for f in train_files])
        cfg_files_eval.extend([os.path.join(map_dir, f) for f in block_eval_files])

    # Agregar los mapas de evaluación fijos con la ruta completa
    cfg_files_eval.extend([os.path.join(map_dir, f) for f in eval_files])

    # Mostrar los resultados
    print("Mapas seleccionados para evaluación fija:", eval_files)
    print("Archivos de configuración para entrenamiento:", cfg_files_train)
    print("Archivos de configuración para evaluación:", cfg_files_eval)

    NUM_EPISODES = 1 # Number of times to run the game. Game restarts with each new episode.
    MAX_AGENT_UPDATES = 30 # Number of times to update (optimize parameters) the bot within an episode.
    NUM_STEPS_POLICY_UPDATE = 12 # Number of experiences to collect for each update to the bot.
    MAX_TOTAL_UPDATES = NUM_EPISODES * MAX_AGENT_UPDATES
    TRAIN = True # Whether to run training or evaluation
    NUM_ENVS = 1 # Number of games to run at once. 
    MAX_EVALUATION_ROUNDS = 1000 # Number of rounds in a game to evaluate the bot.
    USE_SAVED_MODEL = True # Whether to start training or evaluation from a previously saved model.
    MODEL_FILENAME = "ppo_mlp_test.pth" # Name of saved model to start training from and/or to save model to during training.
    STATE_MAPS = True # Set to True to use the state format of maps and architecture CNN and set to False for vector format and architecture MLP
    
    NUM_EPISODES = 10  # Incrementar el número de episodios para entrenar más a fondo.
    MAX_AGENT_UPDATES = 10  # Aumentar el número de actualizaciones del agente por episodio.
    NUM_STEPS_POLICY_UPDATE = 128  # Incrementar los pasos de actualización para una mayor estabilidad.
    NUM_ENVS = 15  # Ajustar según la capacidad de tu máquina. Más entornos aumentan la eficiencia del entrenamiento.
    MAX_EVALUATION_ROUNDS = 1000  # Evaluar durante más rondas para obtener una evaluación precisa.
    USE_SAVED_MODEL = True #Entrenar desde cero o cambiar a True si tienes un modelo preentrenado.
    MODEL_FILENAME = "ppo_mlp_long_training.pth"  # Cambiar el nombre del archivo para el modelo entrenado.
    STATE_MAPS = False  # Mantener si tu entrada son mapas.

    #######################################################################
    # Total number of rounds = MAX_AGENT_UPATES * NUM_STEPS_POLICY_UPDATE #
    #######################################################################

    bots = [
            PPO(state_maps=STATE_MAPS,
             num_envs=NUM_ENVS,
             num_steps=NUM_STEPS_POLICY_UPDATE,
             num_updates=MAX_TOTAL_UPDATES,
             train=TRAIN,
             model_filename = MODEL_FILENAME,
             use_saved_model=USE_SAVED_MODEL),
            PPO(state_maps=STATE_MAPS,
             num_envs=NUM_ENVS,
             num_steps=NUM_STEPS_POLICY_UPDATE,
             num_updates=MAX_TOTAL_UPDATES,
             train=TRAIN,
             model_filename = MODEL_FILENAME,
             use_saved_model=USE_SAVED_MODEL),
            PPO(state_maps=STATE_MAPS,
             num_envs=NUM_ENVS,
             num_steps=NUM_STEPS_POLICY_UPDATE,
             num_updates=MAX_TOTAL_UPDATES,
             train=TRAIN,
             model_filename = MODEL_FILENAME,
             use_saved_model=USE_SAVED_MODEL),
            PPO(state_maps=STATE_MAPS,
             num_envs=NUM_ENVS,
             num_steps=NUM_STEPS_POLICY_UPDATE,
             num_updates=MAX_TOTAL_UPDATES,
             train=TRAIN,
             model_filename = MODEL_FILENAME,
             use_saved_model=USE_SAVED_MODEL),
            PPO(state_maps=STATE_MAPS,
             num_envs=NUM_ENVS,
             num_steps=NUM_STEPS_POLICY_UPDATE,
             num_updates=MAX_TOTAL_UPDATES,
             train=TRAIN,
             model_filename = MODEL_FILENAME,
             use_saved_model=USE_SAVED_MODEL)
             ]
    
    if TRAIN:
        for i in range(1, NUM_EPISODES+1):
            for j in range(len(cfg_files_train)):
                config = engine.GameConfig(cfg_files_train[j])
                game = [engine.Game(config, len(bots)) for i in range(NUM_ENVS)]

                iface = train.Interface(game, bots, debug=False)
                iface.train(max_updates=MAX_AGENT_UPDATES, num_steps_update=NUM_STEPS_POLICY_UPDATE)
    
    if not TRAIN:
        for i in range(1, NUM_EPISODES+1):
            config = engine.GameConfig(cfg_file_eval)
            game = [engine.Game(config, len(bots))]

            iface = train.Interface(game, bots, debug=False)
            iface.run(max_rounds=MAX_EVALUATION_ROUNDS)
            final_scores_list = []
            for bot in bots:
                bot.final_scores_list.append(bot.scores[-1][0])
        
        final_scores = pd.DataFrame()
        for bot in bots:
                final_scores["bot_"+str(bot.player_num)] = bot.final_scores_list

        os.makedirs('./artifacts/outputs', exist_ok=True)
        final_scores.to_csv(f'./artifacts/outputs/{MODEL_FILENAME}.csv', index_label='episode')
        bots[0].save_trained_model()
