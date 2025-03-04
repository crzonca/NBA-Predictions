import itertools
import json
import math
import statistics
import warnings
from datetime import datetime, timedelta

import choix
import networkx as nx
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score


def get_bayes_avg(prior_avg, prior_var, sample_avg, sample_var, n):
    k_0 = sample_var / prior_var
    if k_0 == 0 or n == 0:
        return prior_avg
    posterior_avg = ((k_0 / (k_0 + n)) * prior_avg) + ((n / (k_0 + n)) * sample_avg)
    return posterior_avg


def get_color(value, mean=0, variance=np.nan, alpha=.05, enabled=True, invert=False):
    if not enabled:
        return value

    green = '\033[32m'
    red = '\033[31m'
    stop = '\033[0m'

    if pd.isna(variance):
        value = round(value, 3)
        return red + str(value) + stop if value < 0 else green + str(value) + stop if value > 0 else value
    elif variance == 0:
        return ''
    else:
        normal = norm(mean, math.sqrt(variance))
        if (not invert and normal.ppf(alpha) > value) or (invert and normal.ppf(1 - alpha) < value):
            return red
        if (not invert and normal.ppf(1 - alpha) < value) or (invert and normal.ppf(alpha) > value):
            return green
        return ''


def process_boxscore(boxscore_json):
    team_boxscore = pd.json_normalize(boxscore_json)
    team_boxscore = team_boxscore.drop(columns=['statistics'])
    team_statistics = boxscore_json.get('statistics')
    if not team_statistics:
        return team_boxscore
    team_fgm_fga = [team_stat.get('displayValue') for team_stat in team_statistics if team_stat.get('label') == 'FG'][0]
    team_3pm_3pa = [team_stat.get('displayValue') for team_stat in team_statistics if team_stat.get('label') == '3PT'][0]
    team_ftm_fta = [team_stat.get('displayValue') for team_stat in team_statistics if team_stat.get('label') == 'FT'][0]
    team_oreb = [team_stat.get('displayValue') for team_stat in team_statistics if team_stat.get('label') == 'Offensive Rebounds'][0]
    team_to = [team_stat.get('displayValue') for team_stat in team_statistics if team_stat.get('label') == 'Total Turnovers'][0]

    team_fgm, team_fga = team_fgm_fga.split('-')
    team_3pm, team_3pa = team_3pm_3pa.split('-')
    team_ftm, team_fta = team_ftm_fta.split('-')
    team_points = int(team_fgm) * 2 + int(team_3pm) + int(team_ftm)
    team_possessions = int(team_fga) + int(team_fta) / 2 + int(team_to) - int(team_oreb)
    team_boxscore.at[0, 'FGM'] = int(team_fgm)
    team_boxscore.at[0, 'FGA'] = int(team_fga)
    team_boxscore.at[0, '3FM'] = int(team_3pm)
    team_boxscore.at[0, '3FA'] = int(team_3pa)
    team_boxscore.at[0, 'FTM'] = int(team_ftm)
    team_boxscore.at[0, 'FTA'] = int(team_fta)
    team_boxscore.at[0, 'OREB'] = int(team_oreb)
    team_boxscore.at[0, 'TO'] = int(team_to)
    team_boxscore.at[0, 'Points'] = int(team_points)
    team_boxscore.at[0, 'Possessions'] = int(team_possessions)

    return team_boxscore


class Helper:
    def __init__(self, team_df, individual_df, graph, gen_poisson_model):
        self.team_df = team_df
        self.individual_df = individual_df
        self.graph = graph
        self.gen_poisson_model = gen_poisson_model
        with open('resources/config.json', 'r') as f:
            self.config = json.load(f)

    def predict_possessions_team_averages(self, team1, team2):
        average_possessions = self.config.get('preseason_values').get('preseason_avg_possessions')
        if team1 in list(self.individual_df['Team']):
            if team2 in list(self.individual_df['Team']):
                relevant_df = self.individual_df.loc[(self.individual_df['Team'] == team1) |
                                                     (self.individual_df['Team'] == team2)]
                if not relevant_df.empty:
                    average_possessions = relevant_df['Possessions'].mean()
        else:
            if not self.individual_df.empty:
                average_possessions = self.individual_df['Possessions'].mean()
        return average_possessions

    def predict_possessions(self, team1, team2):
        if not self.team_df.at[team1, 'Possessions Model Good']:
            return self.predict_possessions_team_averages(team1, team2)

        intercept = self.team_df['Possessions Intercept'].mean()
        team1_off_coef = self.team_df.at[team1, 'Off Possessions Coef']
        team1_def_coef = self.team_df.at[team1, 'Def Possessions Coef']
        team2_off_coef = self.team_df.at[team2, 'Off Possessions Coef']
        team2_def_coef = self.team_df.at[team2, 'Def Possessions Coef']

        team1_possessions = intercept + team1_off_coef + team2_def_coef
        team2_possessions = intercept + team2_off_coef + team1_def_coef
        average_possessions = (team1_possessions + team2_possessions) / 2
        return average_possessions

    def get_dist_from_gen_poisson_model(self, team1, team2):
        predicted_possessions = self.predict_possessions(team1, team2)

        explanatory = pd.get_dummies(self.individual_df[['Team', 'Opponent']], dtype=int)
        explanatory = sm.add_constant(explanatory)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            prediction_series = pd.Series(index=explanatory.columns)
            prediction_series.at['const'] = 1.0
            prediction_series.at['Team_' + team1] = 1.0
            prediction_series.at['Opponent_' + team2] = 1.0
            prediction_series = prediction_series.fillna(0.0)
            dist = self.gen_poisson_model.get_distribution(prediction_series, exposure=predicted_possessions)
        return dist

    def predict_score_from_gen_poisson_model(self, team1, team2):
        team1_dist = self.get_dist_from_gen_poisson_model(team1, team2)
        team2_dist = self.get_dist_from_gen_poisson_model(team2, team1)

        # TODO Possibly the following code
        # self.gen_poisson_model.predict(prediction_series, exposure=predicted_possessions)

        return float(team1_dist.mean()), float(team2_dist.mean())

    def gen_poisson_to_sim_skellam_pmf(self, team1_dist, team2_dist, x):
        max_points = self.config.get('betting_constants').get('max_possible_points')
        team2_score_chances = team2_dist.pmf([score for score in range(max_points)])
        team1_score_chances = team1_dist.pmf([score + x for score in range(max_points)])
        return sum(team1_score_chances * team2_score_chances)

    def get_bt_chance(self, team1, team2):
        team1_bt = self.team_df.at[team1, 'Bayes BT']
        team2_bt = self.team_df.at[team2, 'Bayes BT']

        return math.exp(team1_bt) / (math.exp(team1_bt) + math.exp(team2_bt))

    def get_bayes_avg_wins(self, game_df, team_name):
        matching_games = game_df.loc[game_df['Team'] == team_name]

        prior_avg = self.config.get('bayes_priors').get('bayes_wins_prior_mean')
        prior_var = self.config.get('bayes_priors').get('bayes_wins_prior_var')

        wins = list(matching_games['Win'])
        if len(wins) < 2:
            return prior_avg

        win_pct = statistics.mean(wins)
        win_var = statistics.variance(wins)

        return get_bayes_avg(prior_avg, prior_var, win_pct, win_var, len(wins))

    def get_bayes_avg_points(self, game_df, team_name, allowed=False):
        matching_games = game_df.loc[game_df['Team'] == team_name]

        prior_avg = self.config.get('bayes_priors').get('bayes_points_prior_mean')
        prior_var = self.config.get('bayes_priors').get('bayes_points_prior_var')

        points = list(matching_games['Points Allowed']) if allowed else list(matching_games['Points'])
        if len(points) < 2:
            return prior_avg

        avg_points = statistics.mean(points)
        points_var = statistics.variance(points)

        return get_bayes_avg(prior_avg, prior_var, avg_points, points_var, len(points))

    def get_bradley_terry_from_graph(self):
        nodes = self.graph.nodes
        df = pd.DataFrame(nx.to_numpy_array(self.graph), columns=nodes)
        df.index = nodes

        teams = list(df.index)
        df = df.fillna(0)

        teams_to_index = {team: i for i, team in enumerate(teams)}
        index_to_teams = {i: team for team, i in teams_to_index.items()}

        graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        edges = [list(itertools.repeat((teams_to_index.get(team2),
                                        teams_to_index.get(team1)),
                                       int(weight_dict.get('weight'))))
                 for team1, team2, weight_dict in graph.edges.data()]
        edges = list(itertools.chain.from_iterable(edges))

        if not edges:
            coef_df = pd.DataFrame(columns=['BT', 'Var'], index=self.team_df.index)
            coef_df['BT'] = coef_df['BT'].fillna(0)
            coef_df['Var'] = coef_df['Var'].fillna(1)
            return coef_df

        coeffs, cov = choix.ep_pairwise(n_items=len(teams), data=edges, alpha=1)
        coeffs = pd.Series(coeffs)
        cov = pd.Series(cov.diagonal())
        coef_df = pd.DataFrame([coeffs, cov]).T
        coef_df.columns = ['BT', 'Var']
        coef_df.index = [index_to_teams.get(index) for index in coef_df.index]
        coef_df = coef_df.sort_values(by='BT', ascending=False)
        return coef_df

    def get_tiers(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            bts = self.team_df['Bayes BT']
            x = bts.values.reshape(-1, 1)

            tier_count_dict = dict()
            for potential_num_tiers in range(self.config.get('plotting_values').get('min_team_tiers'),
                                             self.config.get('plotting_values').get('max_team_tiers')):
                k_means = KMeans(n_clusters=potential_num_tiers).fit(x)
                tier_count_dict[potential_num_tiers] = calinski_harabasz_score(x, k_means.labels_)

            sorted_tier_counts = list(sorted(tier_count_dict.items(), key=lambda t: t[1], reverse=True))
            num_tiers = sorted_tier_counts[0][0]
            k_means = KMeans(n_clusters=num_tiers).fit(x)

            self.team_df['Cluster'] = k_means.labels_

            cluster_averages = self.team_df.groupby('Cluster').mean(numeric_only=True)
            cluster_averages = cluster_averages.sort_values(by='Bayes BT', ascending=False)
            tiers = {cluster: tier for cluster, tier in zip(cluster_averages.index, range(num_tiers, 0, -1))}
            self.team_df['Tier'] = self.team_df['Cluster'].map(tiers)

            return {team: row['Tier'] for team, row in self.team_df.iterrows()}

    def get_past_games(self, verbose=False):
        events_endpoint = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
        start_date = datetime.strptime(self.config.get('season_start_date'), '%Y-%m-%d')
        query_date = start_date
        # start_date = start_date.strftime('%Y%m%d')

        end_date = datetime.today() + timedelta(days=1)
        # end_date = end_date.strftime('%Y%m%d')

        game_ids = set()
        while query_date < end_date:
            events_params = {'limit': 1500,
                             'dates': query_date.strftime('%Y%m%d')}
                             # 'dates': start_date + '-' + end_date}

            resp = requests.get(events_endpoint, params=events_params).json()

            events_df = pd.json_normalize(resp.get('events'))
            if events_df.empty:
                query_date = query_date + timedelta(days=1)
                continue
            if verbose:
                print('# of Games on', query_date.strftime('%Y-%m-%d'), len(events_df))
            events_df['date'] = pd.to_datetime(events_df['date'])
            events_df = events_df.loc[events_df['status.type.name'] == 'STATUS_FINAL']
            # events_df = events_df.loc[events_df['date'] <= datetime.now(tz=timezone.utc)]

            game_ids = game_ids.union(set(events_df['id']))
            query_date = query_date + timedelta(days=1)

        if verbose:
            print('Total number of Games:', len(game_ids))

        boxscores = list()
        boxscore_endpoint = 'https://cdn.espn.com/core/nba/boxscore'
        for game_id in game_ids:
            if verbose:
                print('Getting boxscore for game', game_id)
            boxscore_params = {'xhr': 1,
                               'gameId': game_id}

            resp = requests.get(boxscore_endpoint, params=boxscore_params).json()
            boxscore = resp.get('gamepackageJSON').get('boxscore').get('teams')

            away_boxscore = process_boxscore(boxscore[0])
            home_boxscore = process_boxscore(boxscore[1])

            away_boxscore.at[0, 'game_id'] = game_id
            home_boxscore.at[0, 'game_id'] = game_id

            away_boxscore.at[0, 'Opponent'] = home_boxscore.at[0, 'team.name']
            home_boxscore.at[0, 'Opponent'] = away_boxscore.at[0, 'team.name']

            boxscores.append(away_boxscore)
            boxscores.append(home_boxscore)

        boxscore_df = pd.concat(boxscores)
        boxscore_df = boxscore_df.rename(columns={'team.name': 'Team',
                                                  'homeAway': 'Location'})
        boxscore_df['Location'] = boxscore_df['Location'].map(str.title)
        boxscore_df.to_csv(self.config.get('output_locations').get('full_boxscore'), index=False)

        return boxscore_df

    def get_past_games2(self):
        domain = 'https://api3.natst.at'
        api_key = '87ab-ba0b1b'

        endpoint = 'teamperfs/nba'

        end_date = datetime.today().astimezone()
        end_date = end_date.strftime('%Y-%m-%d')
        date_range = self.config.get('season_start_date') + ',' + end_date
        resp = requests.get('/'.join([domain, api_key, endpoint, date_range])).json()

        all_games = dict()
        all_games.update(resp.get('performances'))
        while resp.get('meta').get('page-next'):
            resp = requests.get(resp.get('meta').get('page-next')).json()
            all_games.update(resp.get('performances'))

        all_games_df = pd.json_normalize(all_games.values())
        all_games_df.to_csv(self.config.get('output_locations').get('full_boxscore'))

        all_games_df = all_games_df.loc[(all_games_df['game.winorloss'] == 'L') |
                                        (all_games_df['game.winorloss'] == 'W')]

        all_games_df = all_games_df[['game.id', 'team.name', 'opponent.name', 'gameday', 'game.home',
                                     'stats.pts', 'stats.fga', 'stats.fta', 'stats.oreb', 'stats.to']]

        all_games_df = all_games_df.rename(columns={'game.id': 'game_id',
                                                    'team.name': 'Team',
                                                    'opponent.name': 'Opponent',
                                                    'game.home': 'Home Team',
                                                    'stats.pts': 'Points',
                                                    'stats.fga': 'FGA',
                                                    'stats.fta': 'FTA',
                                                    'stats.oreb': 'OREB',
                                                    'stats.to': 'TO'})

        def get_location(r):
            home = r['Home Team']
            team = r['Team']
            return 'Home' if home == team else 'Away'

        all_games_df['Location'] = all_games_df.apply(lambda r: get_location(r), axis=1)
        all_games_df = all_games_df.drop(columns=['Home Team'])

        # TODO We need to find and update the real values
        all_games_df = all_games_df.fillna(-1)

        all_games_df = all_games_df.astype({'Points': 'int32',
                                            'FGA': 'int32',
                                            'FTA': 'int32',
                                            'OREB': 'int32',
                                            'TO': 'int32'})

        all_games_df['Team'] = all_games_df['Team'].map(self.config.get('name_map'))
        all_games_df['Opponent'] = all_games_df['Opponent'].map(self.config.get('name_map'))
        all_games_df['Possessions'] = all_games_df.apply(lambda r: r['FGA'] - r['OREB'] + r['TO'] + int(r['FTA'] / 2),
                                                         axis=1)

        return all_games_df


def get_college_games():
    events_endpoint = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
    start_date = datetime.strptime("2024-11-23", '%Y-%m-%d')
    query_date = start_date
    # start_date = start_date.strftime('%Y%m%d')

    end_date = datetime.today() + timedelta(days=1)
    # end_date = end_date.strftime('%Y%m%d')

    game_ids = set()
    while query_date < end_date:
        events_params = {'limit': 1500,
                         'dates': query_date.strftime('%Y%m%d')}
        # 'dates': start_date + '-' + end_date}

        resp = requests.get(events_endpoint, params=events_params).json()
        query_date = query_date + timedelta(days=1)

        events_df = pd.json_normalize(resp.get('events'))
        print('# of Games on', query_date.strftime('%Y-%m-%d'), len(events_df))
        if events_df.empty:
            continue
        events_df['date'] = pd.to_datetime(events_df['date'])
        events_df = events_df.loc[events_df['status.type.name'] == 'STATUS_FINAL']
        # events_df = events_df.loc[events_df['date'] <= datetime.now(tz=timezone.utc)]

        game_ids = game_ids.union(set(events_df['id']))

    print('Total number of Games:', len(game_ids))
    with open('output/good_bets.txt', 'w') as f:
        f.write('\n'.join(game_ids))