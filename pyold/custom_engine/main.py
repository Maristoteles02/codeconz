#!/usr/bin/python
# -*- coding: utf-8 -*-

import engine, train
import pandas as pd
import os

from bots.randbot import RandBot
from bots.reinforce import REINFORCE


# ==============================================================================
# MAIN
# Main process for simulating matches of different types of bots
# ==============================================================================

if __name__ == "__main__":
    # Set the map
    cfg_file = "maps/grid.txt"
    
    # Set the bots to play the game
    bots = [REINFORCE(state_maps=False, model_filename='reinforce_mlp.pth', use_saved_model=True), 
            REINFORCE(state_maps=True, model_filename='reinforce_cnn.pth', use_saved_model=True), 
            RandBot()]

    NUM_TRAINING_EPISODES = 5
    MAX_ROUNDS = 100

    for i in range(1, NUM_TRAINING_EPISODES+1):
        config = engine.GameConfig(cfg_file)
        game = engine.Game(config, len(bots))

        iface = train.Interface(game, bots, debug=False)
        iface.run(max_rounds=MAX_ROUNDS)

        for bot in bots:
            if bot.save_model:
                bot.save_trained_model()
                print("model saved")
                if bot.last_episode_score < bot.scores[-1]:
                    bot.save_best_model()
                    print("best model saved")
            if bot.last_episode_score < bot.scores[-1]:
                bot.last_episode_score = bot.scores[-1]
            bot.final_scores_list.append(bot.scores[-1])
    
    final_scores = pd.DataFrame()
    rewards_actions_list = pd.DataFrame()
    for bot in bots:
        final_scores[bot.player_num] = bot.final_scores_list
        rewards_actions_list[str(bot.player_num)+"_actions"] = bot.actions_list
        rewards_actions_list[str(bot.player_num)+"_rewards"] = bot.rewards_list
    os.makedirs('./final_scores', exist_ok=True)
    os.makedirs('./actions_rewards', exist_ok=True)
    final_scores.to_csv('./final_scores/final_scores.csv', index_label='episode')
    rewards_actions_list.to_csv('./actions_rewards/rewards_actions_list.csv', index_label='episode')
       