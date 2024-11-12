#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd

import engine.engine as engine
import train as train

from bots.ppo import PPO
from bots.randbot import RandBot


if __name__ == "__main__":
    # Map for training the bot. This can be one map or a list of maps that will 
    cfg_files_train = ["./maps/map_43x23.txt"] 
    # Map for evaluating the bot.
    cfg_file_eval = "./maps/map_43x23.txt"

    NUM_EPISODES = 10 # Number of times to run the game. Game restarts with each new episode.
    MAX_AGENT_UPDATES = 90 # Number of times to update (optimize parameters) the bot within an episode.
    NUM_STEPS_POLICY_UPDATE = 128 # Number of experiences to collect for each update to the bot.
    MAX_EVALUATION_ROUNDS = 1000 # Number of rounds in a game to evaluate the bot.
    TRAIN = True # Whether to run training or evaluation
    NUM_ENVS = 4 # Number of games to run at once. 
    USE_SAVED_MODEL = False # Whether to start training or evaluation from a previously saved model.
    MODEL_FILENAME = "ppo_cnn_test.pth" # Name of saved model to start training from and/or to save model to during training.
    STATE_MAPS = True
    MAX_TOTAL_UPDATES = NUM_EPISODES * MAX_AGENT_UPDATES

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
             use_saved_model=USE_SAVED_MODEL), RandBot()]

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

        os.makedirs('./artifacts/final_scores', exist_ok=True)
        final_scores.to_csv('./artifacts/final_scores/final_scores_cnn_1env_extra_connect3_update.csv', index_label='episode')
