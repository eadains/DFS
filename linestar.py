from bs4 import BeautifulSoup
import quopri
import pandas as pd
import numpy as np


def extract_row_data(row, dtype="current"):
    cells = row.find_all("td")
    if dtype == "current":
        return {
            "Player": cells[5].find(class_="playername").text,
            "Position": cells[4].text,
            # Some teams only have two characters, causing an extra space
            # at the start, so strip that
            "Team": cells[5].find(class_="playerTeam").text[-3:].strip(),
            "Opponent": cells[13].text,
            "Order": cells[15].text,
            "Salary": cells[9].text,
            "Projection": cells[10].find("input").get("value"),
            "Consensus": cells[11].text,
            "pOwn": cells[24].text,
        }
    elif dtype == "historical":
        # TODO: get points actually scored
        return {
            "Player": cells[5].find(class_="playername").text,
            "Position": cells[4].text,
            # Some teams only have two characters, causing an extra space
            # at the start, so strip that
            "Team": cells[5].find(class_="playerTeam").text[-3:].strip(),
            "Salary": cells[9].text,
            "Scored": cells[10].text,
            "Projection": cells[11].find("input").get("value"),
            "Consensus": cells[12].text,
            "Time": cells[13].text,
            "Opponent": cells[14].text,
            "Order": cells[16].text,
            "Bat_Arm": cells[17].text,
            "Consistent": cells[18].text,
            "Floor": cells[19].text,
            "Ceiling": cells[20].text,
            "Avg_FP": cells[22].text,
            "Imp_Runs": cells[23].text,
            "pOwn": cells[25].text,
            "actOwn": cells[26].text,
            "Leverage": cells[27].text,
            "Safety": cells[28].text,
        }


def extract_linestar_data(filename, dtype="current"):
    html = open(filename, "r")
    html = quopri.decodestring(html.read())
    soup = BeautifulSoup(html)

    table = soup.find_all("table")[0]
    row_data = []
    for row in table.find_all("tr", class_="playerCardRow"):
        row_data.append(extract_row_data(row, dtype))

    return pd.DataFrame(row_data)


def make_games(pitchers):
    # Construct unique game strings from pitcher data
    games = set()
    for player in pitchers.itertuples():
        # If @ appears, then the opponent team is at home
        if "@" in player.Opponent:
            # Some teams are only two characters, so selecting the last three
            # causes the @ to be selected as well, so remove it
            game_string = player.Team + "@" + player.Opponent[-3:].replace("@", "")
            games.add(game_string)
        # If @ DOESNT appear, the player is from the at home team
        elif "vs" in player.Opponent:
            # For two character teams, an extra space is selected, so
            # remove it
            game_string = player.Opponent[-3:].strip() + "@" + player.Team
            games.add(game_string)
    return games


def make_game_strings(slate):
    games = make_games(slate[slate["Position"] == "P"])
    game_strings = []
    # For each player, see what game their team belongs to and list them
    for row in slate.itertuples():
        for game in games:
            if row.Team in game:
                game_strings.append(game)
            else:
                continue
    return game_strings


def get_historical_data(date):
    data = extract_linestar_data(f"./data/{date}.mhtml", dtype="historical")
    # Remove (R) and (L) from pitcher names
    data.loc[data["Position"] == "P", "Player"] = data.loc[
        data["Position"] == "P", "Player"
    ].str[:-4]
    data["Salary"] = data["Salary"].replace("[\$,]", "", regex=True).astype(int)
    data["Projection"] = data["Projection"].astype(float)
    data["Scored"] = data["Scored"].astype(float)
    data["Consensus"] = data["Consensus"].astype(float)
    data[["pOwn", "actOwn"]] = (
        data[["pOwn", "actOwn"]].replace("[\%]", "", regex=True).astype(float) / 100
    )
    data["Position"] = data["Position"].str.split("/", expand=True)[0]
    # Replace players with no batting order with NaN
    data["Order"] = data["Order"].replace({"-": np.nan})
    data["Order"] = data["Order"].astype(float)
    data["Game"] = make_game_strings(data)
    data["Opp_Pitcher"] = data.loc[data["Position"] != "P", "Opponent"].str.split(
        ",", expand=True
    )[0]
    return data


def get_proj_data():
    data = extract_linestar_data("./data/proj.mhtml")
    # Remove (R) and (L) from pitcher names
    data.loc[data["Position"] == "P", "Player"] = data.loc[
        data["Position"] == "P", "Player"
    ].str[:-4]
    data["Salary"] = data["Salary"].replace("[\$,]", "", regex=True).astype(int)
    data["Projection"] = data["Projection"].astype(float)
    data["Consensus"] = data["Consensus"].astype(float)
    data["pOwn"] = data["pOwn"].replace("[\%]", "", regex=True).astype(float) / 100
    data["Position"] = data["Position"].str.split("/", expand=True)[0]
    data["Order"] = data["Order"].replace({"-": np.nan})
    data["Order"] = data["Order"].astype(float)
    data["Game"] = make_game_strings(data)
    data["Opp_Pitcher"] = data.loc[data["Position"] != "P", "Opponent"].str.split(
        ",", expand=True
    )[0]
    return data


class OpponentTeams:
    def __init__(self, players):
        self.players = players
        self.position_nums = {
            "P": 1,
            "C/1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "OF": 3,
            "UTIL": 1,
        }
        self.pos_mat = self.get_position_matrix()
        self.teams = pd.get_dummies(players["Team"])
        self.games = pd.get_dummies(players["Game"])

    def get_position_matrix(self):
        """
        Get matrix of 0/1 values indicating whether a player can fill a particular positions
        Each row is a player and each column is a position
        """
        positions = pd.get_dummies(self.players["Position"])
        # If a player plays either C or 1B, they can fill the 1B/C position
        positions["C/1B"] = (
            positions["1B"].astype(bool) | positions["C"].astype(bool)
        ).astype(int)
        positions = positions.drop(columns=["1B", "C"])
        # Reorder columns
        positions = positions[["P", "C/1B", "2B", "3B", "SS", "OF"]]
        return positions

    def position_choice(self, position):
        """
        Chooses a player randomly for given position
        Returns numpy array containing player names selected for position
        """
        # If position is C/1B cathers or first-basemen can fill
        if position == "C/1B":
            data = self.players.loc[
                self.players["Position"].isin(["C", "1B"]), "Player"
            ]
            probs = self.players.loc[self.players["Position"].isin(["C", "1B"]), "pOwn"]
        # If position is UTIL then anyone except pitchers can fill
        elif position == "UTIL":
            data = self.players.loc[self.players["Position"] != "P", "Player"]
            probs = self.players.loc[self.players["Position"] != "P", "pOwn"]
        else:
            data = self.players.loc[self.players["Position"] == position, "Player"]
            probs = self.players.loc[self.players["Position"] == position, "pOwn"]
        # Set probability of choosing each player to the actual ownership numbers normalized
        # to sum to 1
        probs = probs / probs.sum()
        choice = np.random.choice(
            data, self.position_nums[position], replace=False, p=probs
        )
        return choice

    def select_team(self):
        """
        Select entire team
        Returns boolean integer array
        """
        team = []
        for position in self.position_nums.keys():
            team.append(self.position_choice(position))
        team = np.concatenate(team)
        return self.players["Player"].isin(team).astype(int)

    def check_valid_team(self, x):
        """
        Check if a team is valid given FanDuel roster rules
        """
        salary = x @ self.players["Salary"] <= 35000
        min_salary = x @ self.players["Salary"] >= 34000
        teams_con = (x @ self.teams >= 1).sum() >= 3
        games_con = (x @ self.games >= 1).sum() >= 2
        players_con = ((x * (~self.pos_mat["P"].astype(bool))) @ self.teams <= 4).all()
        total = np.sum(x) == 9
        positions_max = [1, 2, 2, 2, 2, 4]
        positions_min = [1, 1, 1, 1, 1, 3]
        positions_max_con = (x @ self.pos_mat <= positions_max).all()
        positions_min_con = (x @ self.pos_mat >= positions_min).all()
        return (
            salary
            & min_salary
            & teams_con
            & games_con
            & players_con
            & total
            & positions_max_con
            & positions_min_con
        )

    def get_team(self):
        while True:
            team = self.select_team()
            if self.check_valid_team(team):
                return team
            else:
                continue

    def get_order_stat(self, points, cutoff, num_opp):
        opp_teams_scores = [self.get_team() @ points for x in range(num_opp)]
        return opp_teams_scores[cutoff - 1]


class ProjectionData:
    def __init__(self):
        self.slate = get_proj_data()
        self.pos_mat = self.get_position_matrix()
        self.teams = pd.get_dummies(self.slate["Team"])
        self.games = pd.get_dummies(self.slate["Game"])

    def get_position_matrix(self):
        """
        Get matrix of 0/1 values indicating whether a player can fill a particular positions
        Each row is a player and each column is a position
        """
        positions = pd.get_dummies(self.slate["Position"])
        # If a player plays either C or 1B, they can fill the 1B/C position
        positions["C/1B"] = (
            positions["1B"].astype(bool) | positions["C"].astype(bool)
        ).astype(int)
        positions = positions.drop(columns=["1B", "C"])
        # Reorder columns
        positions = positions[["P", "C/1B", "2B", "3B", "SS", "OF"]]
        return positions
