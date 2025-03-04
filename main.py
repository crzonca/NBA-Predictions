from app import season as nba

from app import helper
import pandas as pd

import requests
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

if __name__ == '__main__':
    # # get game logs from the reg season
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25',
                                                   league_id_nullable='00',
                                                   season_type_nullable='Regular Season')
    games = gamefinder.get_data_frames()[0]
    # Get a list of distinct game ids
    game_ids = games['GAME_ID'].unique().tolist()
    #
    # headers = {
    #     'Connection': 'keep-alive',
    #     'Accept': 'application/json, text/plain, */*',
    #     'x-nba-stats-token': 'true',
    #     # 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
    #     'x-nba-stats-origin': 'stats',
    #     'Sec-Fetch-Site': 'same-origin',
    #     'Sec-Fetch-Mode': 'cors',
    #     'Referer': 'https://stats.nba.com/',
    #     'Accept-Encoding': 'gzip, deflate, br',
    #     'Accept-Language': 'en-US,en;q=0.9',
    # }
    #
    # # create function that gets pbp logs from the 2020-21 season
    # def get_data(game_id):
    #     play_by_play_url = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_" + game_id + ".json"
    #     response = requests.get(url=play_by_play_url, headers=headers).json()
    #     play_by_play = response['game']['actions']
    #     df = pd.DataFrame(play_by_play)
    #     df['gameid'] = game_id
    #     return df
    #
    # pbpdata = []
    # for game_id in game_ids:
    #     game_data = get_data(game_id)
    #     pbpdata.append(game_data)
    #
    # df = pd.concat(pbpdata, ignore_index=True)
    # df = df.sort_values(by=['gameid', 'orderNumber'])
    # games.to_csv('C:\\Users\\Colin\\OneDrive\\Desktop\\nba_games.csv')
    # df.to_csv('C:\\Users\\Colin\\OneDrive\\Desktop\\nba_pbp.csv')

    nba.season()
