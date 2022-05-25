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
            "Salary": cells[9].text,
            "Projection": cells[10].find("input").get("value"),
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
            "Bat/Arm": cells[17].text,
            "Consistent": cells[18].text,
            "Floor": cells[19].text,
            "Ceiling": cells[20].text,
            "Avg FP": cells[22].text,
            "Imp Runs": cells[23].text,
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
    data[["pOwn", "actOwn"]] = (
        data[["pOwn", "actOwn"]].replace("[\%]", "", regex=True).astype(float)
    )
    data["Position"] = data["Position"].str.split("/", expand=True)[0]
    # Replace players with no batting order with NaN
    data["Order"] = data["Order"].replace({"-": np.nan})
    data["Game"] = make_game_strings(data)
    return data
