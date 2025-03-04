import json
import math
import statistics
import warnings
from datetime import datetime

import networkx as nx
import pandas as pd
import statsmodels.api as sm
from prettytable import PrettyTable
from scipy.stats import norm
from statsmodels.discrete.discrete_model import GeneralizedPoisson

from app import helper as basic
from app.betting import Bettor
from app.evaluation import LeagueEvaluator
from app.helper import Helper
from app.playoff_chances import PlayoffPredictor

with open('resources/config.json', 'r') as config_file:
    config = json.load(config_file)

graph = nx.MultiDiGraph()
gen_poisson_model = None
team_df = pd.DataFrame(columns=config.get('team_df_cols'))
game_df = pd.DataFrame(columns=config.get('game_df_cols'))
individual_df = pd.DataFrame(columns=config.get('individual_df_cols'))


def get_game_results(past_games):
    games_dict = dict()
    for index, row in past_games.iterrows():
        team = row['Team']
        points = row['Points']
        possessions = row['Possessions']
        location = row['Location']
        game_id = row['game_id']

        games_dict[game_id + ' ' + location] = (team, points, possessions, location, game_id)

    game_ids = {game[-1] for key, game in games_dict.items()}
    for game_id in game_ids:
        matching_games = [game for key, game in games_dict.items() if game[-1] == game_id]
        home_version = [game for game in matching_games if game[3] == 'Home'][0]
        away_version = [game for game in matching_games if game[3] == 'Away'][0]

        if game_id != config.get('nba_cup_championship') and game_id not in config.get('all_star_game'):
            set_game_outcome(game_id, home_version[0], away_version[0],
                             home_version[1], away_version[1],
                             home_version[2], away_version[2])

    fit_possessions()
    fit_gen_poisson()
    fit_bt()


def set_game_outcome(game_id, home_name, away_name, home_points, away_points, home_possessions, away_possessions):
    global graph
    global team_df
    global game_df
    global individual_df

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)

    home_victory = home_points > away_points
    away_victory = away_points > home_points

    # Update Game DF
    game_df.loc[len(game_df.index)] = [home_name, 1 if home_victory else 0, home_points, away_points]
    game_df.loc[len(game_df.index)] = [away_name, 1 if away_victory else 0, away_points, home_points]

    # Update Individual DF
    home_game_num = len(individual_df.loc[individual_df['Team'] == home_name]) + 1
    away_game_num = len(individual_df.loc[individual_df['Team'] == away_name]) + 1

    individual_df.loc[len(individual_df.index)] = [game_id, home_name, away_name, home_points, home_possessions,
                                                   home_game_num, 1]
    individual_df.loc[len(individual_df.index)] = [game_id, away_name, home_name, away_points, away_possessions,
                                                   away_game_num, 0]

    winner = home_name if home_victory else away_name
    loser = away_name if home_victory else home_name
    graph.add_edge(loser, winner)

    # Update Team DF
    home_games_played = team_df.at[home_name, 'Games Played']
    away_games_played = team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Games Played'] = home_games_played + 1
    team_df.at[away_name, 'Games Played'] = away_games_played + 1

    team_df.at[home_name, 'Wins'] = team_df.at[home_name, 'Wins'] + 1 if home_victory else team_df.at[home_name, 'Wins']
    team_df.at[away_name, 'Wins'] = team_df.at[away_name, 'Wins'] + 1 if away_victory else team_df.at[away_name, 'Wins']

    team_df.at[home_name, 'Losses'] = team_df.at[home_name, 'Losses'] + 1 \
        if away_victory else team_df.at[home_name, 'Losses']
    team_df.at[away_name, 'Losses'] = team_df.at[away_name, 'Losses'] + 1 \
        if home_victory else team_df.at[away_name, 'Losses']

    team_df.at[home_name, 'Win Pct'] = team_df.at[home_name, 'Wins'] / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Win Pct'] = team_df.at[away_name, 'Wins'] / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Bayes Win Pct'] = helper.get_bayes_avg_wins(game_df, home_name)
    team_df.at[away_name, 'Bayes Win Pct'] = helper.get_bayes_avg_wins(game_df, away_name)

    team_df.at[home_name, 'Avg Points'] = (team_df.at[home_name, 'Avg Points'] * home_games_played
                                           + home_points) / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Avg Points'] = (team_df.at[away_name, 'Avg Points'] * away_games_played
                                           + away_points) / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Bayes Avg Points'] = helper.get_bayes_avg_points(game_df, home_name)
    team_df.at[away_name, 'Bayes Avg Points'] = helper.get_bayes_avg_points(game_df, away_name)

    team_df.at[home_name, 'Avg Points Allowed'] = (team_df.at[home_name, 'Avg Points Allowed'] * home_games_played
                                                   + away_points) / team_df.at[home_name, 'Games Played']
    team_df.at[away_name, 'Avg Points Allowed'] = (team_df.at[away_name, 'Avg Points Allowed'] * away_games_played
                                                   + home_points) / team_df.at[away_name, 'Games Played']

    team_df.at[home_name, 'Bayes Avg Points Allowed'] = helper.get_bayes_avg_points(game_df, home_name, allowed=True)
    team_df.at[away_name, 'Bayes Avg Points Allowed'] = helper.get_bayes_avg_points(game_df, away_name, allowed=True)

    with pd.option_context('future.no_silent_downcasting', True):
        team_df = team_df.fillna(0)


def fit_gen_poisson():
    # If a ~ GP(mu1, alpha) and b ~ GP(mu2, alpha), a + b ~ GP(mu1 + mu2, alpha)
    global team_df
    global gen_poisson_model

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    average_possessions = helper.predict_possessions_team_averages('', '')

    if min(team_df['Games Played']) < 3:
        team_df['Adjusted Points'] = config.get('preseason_values').get('league_average_points')
        team_df['Adjusted Points Allowed'] = config.get('preseason_values').get('league_average_points')
        team_df['Adjusted Point Diff'] = team_df.apply(lambda r: r['Adjusted Points'] - r['Adjusted Points Allowed'],
                                                       axis=1)

        with pd.option_context('future.no_silent_downcasting', True):
            team_df = team_df.fillna(0)
        return

    response = individual_df['Points']
    explanatory = pd.get_dummies(individual_df[['Team', 'Opponent']], dtype=int)
    explanatory = sm.add_constant(explanatory)

    gen_poisson = GeneralizedPoisson(endog=response,
                                     exog=explanatory,
                                     exposure=individual_df['Possessions'],
                                     p=1)

    gen_poisson_model = gen_poisson.fit_regularized(method='l1',
                                                    maxiter=int(1e6),
                                                    alpha=config.get('regression_constants').get('l1_factor'),
                                                    disp=0)

    with open(config.get('output_locations').get('regression_summary'), 'w') as f:
        f.write(str(gen_poisson_model.summary()))
        f.close()

    series_index = explanatory.columns

    for team in team_df.index:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf_prediction_series = pd.Series(index=series_index)
            pf_prediction_series.at['const'] = 1.0
            pf_prediction_series.at['Team_' + team] = 1.0
            pf_prediction_series = pf_prediction_series.fillna(0.0)

            adj_points_for = gen_poisson_model.predict(pf_prediction_series, exposure=average_possessions).squeeze()
            team_df.at[team, 'Adjusted Points'] = adj_points_for

            pa_prediction_series = pd.Series(index=series_index)
            pa_prediction_series.at['const'] = 1.0
            pa_prediction_series.at['Opponent_' + team] = 1.0
            pa_prediction_series = pa_prediction_series.fillna(0.0)

            adj_points_allowed = gen_poisson_model.predict(pa_prediction_series, exposure=average_possessions).squeeze()
            team_df.at[team, 'Adjusted Points Allowed'] = adj_points_allowed

            team_df.at[team, "Off Coef"] = gen_poisson_model.params['Team_' + team]
            team_df.at[team, "Def Coef"] = gen_poisson_model.params['Opponent_' + team]

    team_df['Completeness'] = team_df.apply(lambda r: r['Off Coef'] - r['Def Coef'], axis=1)
    team_df['Adjusted Point Diff'] = team_df.apply(lambda r: r['Adjusted Points'] - r['Adjusted Points Allowed'],
                                                   axis=1)


def fit_bt():
    global graph
    global team_df

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    bt_df = helper.get_bradley_terry_from_graph()

    bts = {index: row['BT'] for index, row in bt_df.iterrows()}
    bt_vars = {index: row['Var'] for index, row in bt_df.iterrows()}

    for team_name in team_df.index:
        team_df.at[team_name, 'BT'] = bts.get(team_name)
        team_df.at[team_name, 'BT Var'] = bt_vars.get(team_name)

        set_bayes_bt(team_name)

    bayes_bts = {index: row['Bayes BT'] for index, row in team_df.iterrows()}
    bt_var = statistics.variance(bayes_bts.values())
    bt_sd = math.sqrt(bt_var)
    bt_norm = norm(0, bt_sd)
    for team_name in team_df.index:
        team_df.at[team_name, 'BT Pct'] = bt_norm.cdf(bayes_bts.get(team_name, 0))

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    tiers = helper.get_tiers()
    for team, tier in tiers.items():
        team_df.at[team, 'Tier'] = tier
    team_df = team_df.drop(columns=['Cluster'])

    with pd.option_context('future.no_silent_downcasting', True):
        team_df = team_df.fillna(0)


def fit_possessions(verbose=False):
    y = individual_df['Possessions']
    x = pd.get_dummies(individual_df[['Team', 'Opponent']], dtype=int)
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()

    if verbose:
        if results.f_pvalue > .05:
            print('OLS model is not a good fit')
        print(results.summary())

    intercept = results.params['const']
    for team, row in team_df.iterrows():
        if team_df.at[team, 'Games Played'] == 0:
            team_df.at[team, 'Possessions Model Good'] = results.f_pvalue < .05
            team_df.at[team, 'Possessions Intercept'] = intercept
            team_df.at[team, 'Off Possessions Coef'] = 0
            team_df.at[team, 'Def Possessions Coef'] = 0
            continue
        team_df.at[team, 'Possessions Model Good'] = results.f_pvalue < .05
        team_df.at[team, 'Possessions Intercept'] = intercept
        team_df.at[team, 'Off Possessions Coef'] = results.params['Team_' + str(team)]
        team_df.at[team, 'Def Possessions Coef'] = results.params['Opponent_' + str(team)]


def make_game_predictions(today):
    with open(config.get('resource_locations').get('schedule'), 'r') as f:
        schedule = json.load(f)
    with open(config.get('resource_locations').get('schedule_dates'), 'r') as f:
        schedule_dates = json.load(f)

    matchups = list()
    for team, opponents in schedule.items():
        opponent_dates = list(zip(opponents, schedule_dates.get(team)))
        opponent = [opponent for opponent, date in opponent_dates if date == datetime.strftime(today, '%Y-%m-%d')]
        if opponent:
            opponent = opponent[0]
            if opponent not in [t for t, o in matchups]:
                matchups.append((team, opponent))

    if not matchups:
        return

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    bets = Bettor(team_df, individual_df, graph, gen_poisson_model)
    scores = list()
    bt_chances = list()
    reg_chances = list()
    for team1, team2 in matchups:
        team1_score, team2_score = helper.predict_score_from_gen_poisson_model(team1, team2)
        team1_higher_score = team1_score > team2_score
        score_winner = team1 if team1_higher_score else team2
        score_loser = team2 if team1_higher_score else team1
        higher_score = team1_score if team1_higher_score else team2_score
        lower_score = team2_score if team1_higher_score else team1_score
        score_diff = higher_score - lower_score
        scores.append((score_winner, score_loser, higher_score, lower_score, score_diff))

        team1_chance = helper.get_bt_chance(team1, team2)
        team1_favored = team1_chance >= .5
        favored = team1 if team1_favored else team2
        underdog = team2 if team1_favored else team1
        favored_chance = team1_chance if team1_favored else 1 - team1_chance
        bt_chances.append((favored, underdog, favored_chance))

        cover_chance, push_chance, fail_chance = bets.get_spread_chance(team1, team2, 0)
        team1_chance = cover_chance / (cover_chance + fail_chance)
        team1_favored = team1_chance >= .5
        favored = team1 if team1_favored else team2
        underdog = team2 if team1_favored else team1
        favored_chance = team1_chance if team1_favored else 1 - team1_chance
        reg_chances.append((favored, underdog, favored_chance))

    scores = sorted(scores, key=lambda t: t[-1], reverse=True)
    score_ljust_1 = max([len(t[0]) for t in scores]) + 1
    score_ljust_2 = max([len(t[1]) for t in scores]) + 1
    for score_winner, score_loser, higher_score, lower_score, score_diff in scores:
        print('The', score_winner.ljust(score_ljust_1), 'are projected to beat the', score_loser.ljust(score_ljust_2),
              'by', str(round(score_diff, 1)).ljust(4),
              '(' + str(round(higher_score)) + ' - ' + str(round(lower_score)) + ')')
    print()

    bt_chances = sorted(bt_chances, key=lambda t: t[-1], reverse=True)
    bt_chance_ljust = max([len(t[0]) for t in bt_chances]) + 1
    for favored, underdog, favored_chance in bt_chances:
        print('The',  favored.ljust(bt_chance_ljust),
              'have a ' + f'{favored_chance * 100:.1f}' + '% chance to beat the', underdog.ljust(15))
    print()

    # reg_chances = sorted(reg_chances, key=lambda t: t[-1], reverse=True)
    # reg_chance_ljust = max([len(t[0]) for t in reg_chances]) + 1
    # for favored, underdog, favored_chance in reg_chances:
    #     print('The',  favored.ljust(reg_chance_ljust),
    #           'have a ' + f'{favored_chance * 100:.1f}' + '% chance to beat the', underdog.ljust(15))
    # print()


def print_table(sort_key='Bayes BT'):
    global team_df

    ascending_order = True if sort_key in config.get('ascending_cols') else False
    team_df = team_df.sort_values(by=sort_key, kind='mergesort', ascending=ascending_order)

    columns = ['Rank', 'Name', 'Record', 'Bayes Win %', 'Score',
               'Proj. Record', 'Adj. PPG', 'Adj. PPG Allowed', 'Adj. Point Diff']

    table = PrettyTable(columns)
    table.float_format = '0.3'

    points_coefs = team_df['Adjusted Points']
    points_allowed_coefs = team_df['Adjusted Points Allowed']

    points_avg = statistics.mean(points_coefs)
    points_allowed_avg = statistics.mean(points_allowed_coefs)
    points_diff_avg = statistics.mean(team_df['Adjusted Point Diff'])

    points_var = statistics.variance(points_coefs)
    points_allowed_var = statistics.variance(points_allowed_coefs)
    points_diff_var = statistics.variance(team_df['Adjusted Point Diff'])

    stop = '\033[0m'
    pp = PlayoffPredictor(team_df, graph)

    for index, row in team_df.iterrows():
        table_row = list()

        wins = row['Wins']
        losses = row['Losses']
        record = ' - '.join([str(int(val)).rjust(2) for val in [wins, losses]])

        rank = team_df.index.get_loc(index) + 1

        points_pct = .1
        points_color = basic.get_color(row['Adjusted Points'], points_avg, points_var, alpha=points_pct)
        points_allowed_color = basic.get_color(row['Adjusted Points Allowed'],
                                               points_allowed_avg,
                                               points_allowed_var,
                                               alpha=points_pct,
                                               invert=True)
        points_diff_color = basic.get_color(row['Adjusted Point Diff'],
                                            points_diff_avg,
                                            points_diff_var,
                                            alpha=points_pct,
                                            invert=False)

        if config.get('sort_by_division'):
            table_row.append(row['Division'])
        else:
            table_row.append(rank)
        table_row.append(index)
        table_row.append(record)
        table_row.append((f"{row['Bayes Win Pct'] * 100:.1f}" + '%').rjust(5))

        bt_color = basic.get_color(row['BT'], 0, row['BT Var'])
        table_row.append(bt_color + f"{row['BT Pct'] * 100:.1f}".rjust(5) + stop)

        proj_record = pp.get_proj_record(index)
        proj_record = ' - '.join([str(val).rjust(2) for val in proj_record])
        table_row.append(proj_record)

        table_row.append(points_color + str(round(row['Adjusted Points'], 1)) + stop)
        table_row.append(points_allowed_color + str(round(row['Adjusted Points Allowed'], 1)) + stop)
        table_row.append(points_diff_color + str(round(row['Adjusted Point Diff'], 1)).rjust(5) + stop)

        table.add_row(table_row)

    # Print the table
    with open(config.get('output_locations').get('rankings'), 'w') as f:
        f.write('Rankings\n')
        table_str = str(table)
        table_str = table_str.replace(stop, '')
        table_str = table_str.replace('\033[32m', '')
        table_str = table_str.replace('\033[31m', '')
        f.write(table_str)
        f.close()

    print('Rankings')
    print(table)
    print()


def set_bayes_bt(team):
    global team_df

    games_played = team_df.at[team, 'Wins'] + team_df.at[team, 'Losses']

    bt_var = team_df.at[team, 'BT Var']
    bt_var = config.get('preseason_values').get('preseason_bt_var') if pd.isna(bt_var) else bt_var

    sample_avg = team_df.at[team, 'BT']
    sample_avg = team_df.at[team, 'Preseason BT'] if pd.isna(sample_avg) else sample_avg

    bayes_bt = basic.get_bayes_avg(team_df.at[team, 'Preseason BT'],
                                   config.get('bayes_priors').get('bayes_bt_prior_var'),
                                   sample_avg,
                                   bt_var,
                                   games_played)

    team_df.at[team, 'Bayes BT'] = bayes_bt


def season():
    global graph
    global team_df
    global game_df

    strings = {'Team'}
    ints = {'Games Played', 'Wins', 'Losses'}
    bools = {'Possessions Model Good'}
    floats = set(team_df.columns) - strings - ints - bools
    dtype_map = {col: 'string' for col in strings}
    dtype_map.update({col: 'int32' for col in ints})
    dtype_map.update({col: 'bool' for col in bools})
    dtype_map.update({col: 'float64' for col in floats})
    team_df = team_df.astype(dtype_map)

    le = LeagueEvaluator(team_df, individual_df, graph, gen_poisson_model)

    teams = pd.Series(config.get('teams'))
    team_df['Team'] = teams
    team_df = team_df.set_index('Team')

    with pd.option_context('future.no_silent_downcasting', True):
        team_df['BT Var'] = team_df['BT Var'].fillna(config.get('preseason_values').get('preseason_bt_var'))
        team_df['Possessions Model Good'] = team_df['Possessions Model Good'].fillna(False)
        team_df = team_df.fillna(0.0)

    today = datetime.today()
    is_preseason = today < datetime.strptime(config.get('season_start_date'), '%Y-%m-%d')
    preseason_bts = le.get_preseason_bts()
    for team, pre_bt in preseason_bts.items():
        team_df.at[team, 'Preseason BT'] = pre_bt
        if is_preseason:
            team_df.at[team, 'Bayes Win Pct'] = config.get('bayes_priors').get('bayes_wins_prior_mean')
        set_bayes_bt(team)

    bayes_bts = {index: row['Bayes BT'] for index, row in team_df.iterrows()}
    bt_var = statistics.variance(bayes_bts.values())
    bt_sd = math.sqrt(bt_var)
    bt_norm = norm(0, bt_sd)
    for team_name in team_df.index:
        team_df.at[team_name, 'BT Pct'] = bt_norm.cdf(bayes_bts.get(team_name, 0))

    if is_preseason:
        print('Preseason')
        print_table()

    helper = Helper(team_df, individual_df, graph, gen_poisson_model)
    previous_games = helper.get_past_games()
    get_game_results(previous_games)

    make_game_predictions(today)

    print_table()

    le = LeagueEvaluator(team_df, individual_df, graph, gen_poisson_model)
    le.show_off_def()
    le.show_graph()

    if min(team_df['Games Played']) >= 3:
        bets = Bettor(team_df, individual_df, graph, gen_poisson_model)
        bets.all_bets(config.get('betting_constants').get('betting_wallet'))

    team_df.to_csv(config.get('output_locations').get('team_df'))
