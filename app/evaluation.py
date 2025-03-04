import json
import math
import statistics
import warnings

import PIL
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

from app import league_structure


class LeagueEvaluator:
    def __init__(self, team_df, individual_df, graph, gen_poisson_model):
        self.team_df = team_df
        self.individual_df = individual_df
        self.graph = graph
        self.gen_poisson_model = gen_poisson_model
        with open('resources/config.json', 'r') as f:
            self.config = json.load(f)

    def get_preseason_bts(self, use_mse=True, use_persisted=True):
        path = self.config.get('resource_locations').get('preseason_bts')
        if use_persisted:
            pre_bt_df = pd.read_csv(path)
            team_bts = {row['Team']: row['BT'] for index, row in pre_bt_df.iterrows()}
            return team_bts
        win_totals = self.config.get('preseason_win_totals')
        win_totals = {team: total - 2 if total > 55 else total for team, total in win_totals.items()}
        win_totals = {team: total - 1 if 55 >= total > 27 else total for team, total in win_totals.items()}

        schedules = league_structure.load_schedule()

        teams = list(win_totals.keys())
        bts = np.zeros(len(teams))
        win_proj = np.array(list(win_totals.values()))

        def objective(params):
            val = np.float64(0)

            for team, opps in schedules.items():
                team_proj = np.float64(0)
                team_index = teams.index(team)
                for opponent in opps:
                    opponent_index = teams.index(opponent)

                    team_proj += 1 / np.exp(np.logaddexp(0, -(params[team_index] - params[opponent_index])))

                if use_mse:
                    val += (win_proj[team_index] - team_proj) ** 2
                else:
                    val += np.abs(win_proj[team_index] - team_proj)

            return val

        res = minimize(objective, bts, method='Powell', jac=False)

        def get_bt_prob(bt1, bt2):
            return math.exp(bt1) / (math.exp(bt1) + math.exp(bt2))

        team_bts = {team: bt for team, bt in zip(teams, res.x)}

        rows = list()
        for team, opponents in schedules.items():
            proj_wins = sum([get_bt_prob(team_bts.get(team), team_bts.get(opponent)) for opponent in opponents])
            diff = proj_wins - win_totals.get(team)

            row = {'Team': team,
                   'BT': team_bts.get(team),
                   'BT Projection': proj_wins,
                   'Odds Projection': win_totals.get(team),
                   'Diff': diff,
                   'Abs Diff': abs(diff)}
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(by='BT', ascending=False)
        df = df.reset_index(drop=True)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)

        df['Rank'] = range(1, 31)
        df = df.set_index('Rank', drop=True)
        df = df[['Team', 'BT', 'Odds Projection', 'BT Projection']]
        df['BT Projection'] = df['BT Projection'].round(1)
        df.to_csv(path, index=False)

        return team_bts

    def show_off_def(self):
        warnings.filterwarnings("ignore")

        sns.set(style="ticks")

        # Format and title the graph
        fig, ax = plt.subplots(figsize=(20, 10))

        ax.set_title('')
        ax.set_xlabel('Adjusted Points For')
        ax.set_ylabel('Adjusted Points Against')
        ax.set_facecolor('#FAFAFA')

        images = {team: PIL.Image.open('resources/logos/' + team + '.png')
                  for team, row in self.team_df.iterrows()}

        margin = 1
        min_x = self.team_df['Adjusted Points'].min() - margin
        max_x = self.team_df['Adjusted Points'].max() + margin

        min_y = self.team_df['Adjusted Points Allowed'].min() - margin
        max_y = self.team_df['Adjusted Points Allowed'].max() + margin

        ax = plt.gca()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y)
        ax.set_aspect(aspect=0.3, adjustable='datalim')

        for team in self.team_df.index:
            xa = self.team_df.at[team, 'Adjusted Points']
            ya = self.team_df.at[team, 'Adjusted Points Allowed']

            offset = .6
            img_alpha = .8  # if team in playoff_teams else .2
            ax.imshow(images.get(team), extent=(xa - offset, xa + offset, ya + offset, ya - offset), alpha=img_alpha)

        vert_mean = self.team_df['Adjusted Points'].mean()
        horiz_mean = self.team_df['Adjusted Points Allowed'].mean()
        plt.axvline(x=vert_mean, color='r', linestyle='--', alpha=.5)
        plt.axhline(y=horiz_mean, color='r', linestyle='--', alpha=.5)

        offset_dist = 5 * math.sqrt(2)
        offsets = set(np.arange(0, 75, offset_dist))
        offsets = offsets.union({-offset for offset in offsets})

        for offset in [horiz_mean + offset for offset in offsets]:
            plt.axline(xy1=(vert_mean, offset), slope=1, alpha=.1)

        # Show the graph
        plt.savefig(self.config.get('output_locations').get('offense_defense'), dpi=300)
        # plt.show()

    def show_graph(self):
        warnings.filterwarnings("ignore")

        sns.set(style="ticks")

        nba = self.graph.copy()

        # Format and title the graph
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_aspect('auto')
        ax.set_title('')
        ax.set_facecolor('#FAFAFA')

        # Get the Pagerank of each node
        bts = {team: row['Bayes BT'] for team, row in self.team_df.iterrows()}
        max_bt = self.team_df['Bayes BT'].max()
        min_bt = self.team_df['Bayes BT'].min()
        bt_dev = statistics.stdev(self.team_df['Bayes BT'])
        subset = {team: row['Tier'] for team, row in self.team_df.iterrows()}

        nx.set_node_attributes(nba, bts, 'Bayes BT')
        nx.set_node_attributes(nba, subset, 'subset')

        images = {team: PIL.Image.open('resources/logos/' + team + '.png')
                  for team, row in self.team_df.iterrows()}
        nx.set_node_attributes(nba, images, 'image')

        pos = nx.multipartite_layout(nba, align='horizontal')

        edge_list = nba.edges()

        # Draw the edges in the graph
        for source, target in edge_list:
            conn_stlye = "Arc3, rad=0.2" if subset.get(source) == subset.get(target) else "Arc3, rad=0.05"
            target_bt = bts.get(target)
            target_margin = math.exp(target_bt) * 18

            nx.draw_networkx_edges(nba,
                                   pos,
                                   edgelist=[(source, target)],
                                   width=1,
                                   alpha=0.02,
                                   edge_color='black',
                                   connectionstyle=conn_stlye,
                                   arrowsize=10,
                                   min_target_margin=target_margin)

        # Select the size of the image (relative to the X axis)
        icon_scale_factor = self.config.get('plotting_values').get('icon_scale_factor')
        individual_icon_scales = self.config.get('team_icon_scale')
        icon_size = {team: (ax.get_xlim()[1] - ax.get_xlim()[0]) * math.exp(bt) * icon_scale_factor *
                           individual_icon_scales.get(team, 1.0) for team, bt in bts.items()}
        icon_center = {team: size / 2.0 for team, size in icon_size.items()}

        for n in nba.nodes:
            xa, ya = fig.transFigure.inverted().transform(ax.transData.transform(pos[n]))
            a = plt.axes([xa - icon_center.get(n), ya - icon_center.get(n), icon_size.get(n), icon_size.get(n)])
            a.set_aspect('auto')
            a.imshow(nba.nodes[n]['image'], alpha=1.0)
            a.axis("off")

        # Show the graph
        plt.savefig(self.config.get('output_locations').get('league_graph'), dpi=300)
        # plt.show()
