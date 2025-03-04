import itertools
import math
import json

import numpy as np

from app import league_structure


class PlayoffPredictor:
    def __init__(self, team_df, graph):
        self.team_df = team_df
        self.graph = graph
        with open('resources/config.json', 'r') as f:
            self.config = json.load(f)

    def get_remaining_win_probs(self, team_name):
        schedule = league_structure.load_schedule()
        opponents = schedule.get(team_name)

        bt = self.team_df.at[team_name, 'Bayes BT']
        wins = self.team_df.at[team_name, 'Wins']
        losses = self.team_df.at[team_name, 'Losses']
        games_played = int(wins + losses)
        if games_played >= self.config.get('regular_season_games'):
            return []

        remaining_opponents = opponents[games_played:self.config.get('regular_season_games')]
        opponent_bts = [self.team_df.at[opponent, 'Bayes BT'] for opponent in remaining_opponents]
        win_probs = [math.exp(bt) / (math.exp(bt) + math.exp(opp_bt)) for opp_bt in opponent_bts]

        return win_probs

    def get_total_wins_chances(self, team):
        wins = self.team_df.at[team, 'Wins']
        wins_dict = {win_total: 0.0 for win_total in range(self.config.get('regular_season_games'))}

        win_probs = self.get_remaining_win_probs(team)
        loss_probs = [1 - win_prob for win_prob in win_probs]

        win_mask = list(itertools.product([0, 1], repeat=len(win_probs)))
        for win_combo in win_mask:
            loss_combo = [0 if game == 1 else 1 for game in win_combo]

            win_combo_probs = list(itertools.compress(win_probs, win_combo))
            loss_combo_probs = list(itertools.compress(loss_probs, loss_combo))
            win_combo_wins = len(win_combo_probs) + wins

            total_wins_prob = np.prod(win_combo_probs)
            total_losses_prob = np.prod(loss_combo_probs)

            combo_prob = total_wins_prob * total_losses_prob

            wins_dict[win_combo_wins] = wins_dict.get(win_combo_wins) + combo_prob

        return wins_dict

    def get_proj_record(self, team_name):
        win_probs = self.get_remaining_win_probs(team_name)

        wins = self.team_df.at[team_name, 'Wins']

        expected_wins = sum(win_probs) + wins
        expected_losses = self.config.get('regular_season_games') - expected_wins

        # TODO Consider Removing
        missing_games = 82 - self.config.get('regular_season_games')
        expected_wp = expected_wins / (expected_wins + expected_losses)
        missing_wins = missing_games * expected_wp
        missing_losses = missing_games * (1 - expected_wp)

        return round(expected_wins + missing_wins), round(expected_losses + missing_losses)
