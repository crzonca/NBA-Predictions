import json
import time

import pandas as pd
from sportsipy.nba.schedule import Schedule
from sportsipy.nba.teams import Teams

with open('resources/config.json', 'r') as config_file:
    config = json.load(config_file)


def get_name_from_abbrev(abbrev):
    abbrev_to_name = config.get('team_abbreviations')

    return abbrev_to_name.get(abbrev, abbrev)


def load_schedule():
    with open(config.get('resource_locations').get('schedule'), 'r') as f:
        schedule = json.load(f)
        return schedule


# TODO Change to get previous games
def get_games_before_week(today, use_persisted=True):
    if use_persisted:
        week_results = pd.read_csv(config.get('resource_locations').get('games'))
        week_results = week_results.dropna()
    else:
        teams = Teams()
        games_in_week = list()
        for abbrev in teams.dataframes['abbreviation']:
            sch = Schedule(abbrev).dataframe
            sch['team'] = abbrev
            game = sch.loc[sch['week'] <= today]
            game = game.reset_index(drop=True)
            if not game.empty and game.loc[0]['points_scored'] is not None:
                games_in_week.append(game)
            time.sleep(5)
        if games_in_week:
            week_results = pd.concat(games_in_week)
        else:
            week_results = pd.DataFrame()
    return week_results
