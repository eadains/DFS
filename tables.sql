CREATE TABLE Stadiums (
    StadiumID PRIMARY KEY,
    Active,
    Name,
    City,
    State,
    Country,
    Capacity,
    Surface,
    LeftField,
    MidLeftField,
    LeftCenterField,
    MidLeftCenterField,
    CenterField,
    MidRightCenterField,
    RightCenterField,
    MidRightField,
    RightField,
    GeoLat,
    GeoLong,
    Altitude,
    HomePlateDirection,
    Type
);

CREATE TABLE Teams (
    TeamID PRIMARY KEY,
    Key,
    Active,
    City,
    Name,
    StadiumID REFERENCES Stadiums(StadiumID),
    League,
    Division,
    PrimaryColor,
    SecondaryColor,
    TertiaryColor,
    QuaternaryColor,
    WikipediaLogoUrl,
    WikipediaWordMarkUrl,
    GlobalTeamID
);

CREATE TABLE Players (
    PlayerID PRIMARY KEY,
    Status,
    TeamID REFERENCES Teams(TeamID),
    Team,
    Jersey,
    PositionCategory,
    Position,
    FirstName,
    LastName,
    BatHand,
    ThrowHand,
    Height,
    Weight,
    BirthDate,
    BirthCity,
    BirthState,
    BirthCountry,
    PhotoUrl,
    InjuryStatus,
    FanDuelPlayerID,
    DraftKingsPlayerID,
    FanDuelName,
    DraftKingsName
);

CREATE TABLE Games (
    GameID PRIMARY KEY,
    Season,
    SeasonType,
    Status,
    Day,
    DateTime,
    AwayTeam,
    HomeTeam,
    AwayTeamID REFERENCES Teams(TeamID),
    HomeTeamID REFERENCES Teams(TeamID),
    StadiumID REFERENCES Stadiums(StadiumID),
    AwayTeamRuns,
    HomeTeamRuns,
    AwayTeamProbablePitcherID REFERENCES Players(PlayerID),
    HomeTeamProbablePitcherID REFERENCES Players(PlayerID),
    AwayTeamStartingPitcherID REFERENCES Players(PlayerID),
    HomeTeamStartingPitcherID REFERENCES Players(PlayerID),
    PointSpread,
    OverUnder,
    AwayTeamMoneyLine,
    HomeTeamMoneyLine,
    ForecastTempLow,
    ForecastTempHigh,
    ForecastDescription,
    ForecastWindChill,
    ForecastWindSpeed,
    ForecastWindDirection,
    AwayTeamStartingPitcher,
    HomeTeamStartingPitcher,
    HomeRotationNumber,
    AwayRotationNumber,
    NeutralVenue,
    OverPayout,
    UnderPayout
);

CREATE TABLE PlayerStats (
    StatID,
    TeamID REFERENCES Teams(TeamID),
    PlayerID REFERENCES Players(PlayerID),
    SeasonType,
    Season,
    Name,
    Team,
    Position,
    PositionCategory,
    Started,
    InjuryStatus,
    GameID REFERENCES Games(GameID),
    OpponentID REFERENCES Teams(TeamID),
    Opponent,
    Day,
    DateTime,
    HomeOrAway,
    Games,
    FantasyPoints,
    AtBats,
    Runs,
    Hits,
    Singles,
    Doubles,
    Triples,
    HomeRuns,
    RunsBattedIn,
    BattingAverage,
    Outs,
    Strikeouts,
    Walks,
    HitByPitch,
    Sacrifices,
    SacrificeFlies,
    GroundIntoDoublePlay,
    StolenBases,
    CaughtStealing,
    OnBasePercentage,
    SluggingPercentage,
    OnBasePlusSlugging,
    Wins,
    Losses,
    Saves,
    InningsPitchedDecimal,
    TotalOutsPitched,
    InningsPitchedFull,
    InningsPitchedOuts,
    EarnedRunAverage,
    PitchingHits,
    PitchingRuns,
    PitchingEarnedRuns,
    PitchingWalks,
    PitchingStrikeouts,
    PitchingHomeRuns,
    PitchesThrown,
    PitchesThrownStrikes,
    WalksHitsPerInningsPitched,
    PitchingBattingAverageAgainst,
    FantasyPointsFanDuel,
    FantasyPointsDraftKings,
    WeightedOnBasePercentage,
    PitchingCompleteGames,
    PitchingShutOuts,
    PitchingOnBasePercentage,
    PitchingSluggingPercentage,
    PitchingOnBasePlusSlugging,
    PitchingStrikeoutsPerNineInnings,
    PitchingWalksPerNineInnings,
    PitchingWeightedOnBasePercentage
);

CREATE TABLE Slate (
    Date,
    GameID REFERENCES Games(GameID),
    PlayerID REFERENCES Players(PlayerID),
    TeamID REFERENCES Teams(TeamID),
    Name,
    Position,
    Salary
);