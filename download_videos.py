import os
from SoccerNet.Downloader import SoccerNetDownloader

PASSWORD = "s0cc3rn3t"

downloader = SoccerNetDownloader(LocalDirectory="data/soccernet")
downloader.password = PASSWORD

#list of games you desire to download
games = [
    # --- already downloaded (EPL) ---
    "england_epl/2016-2017/2016-10-02 - 18-30 Burnley 0 - 1 Arsenal",
    "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea",
    "england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool",
    "england_epl/2015-2016/2015-11-29 - 15-00 Tottenham 0 - 0 Chelsea",

    # --- new batch ---
    "germany_bundesliga/2016-2017/2017-04-29 - 16-30 Dortmund 0 - 0 FC Koln",
    "germany_bundesliga/2015-2016/2015-11-07 - 17-30 Bayern Munich 4 - 0 VfB Stuttgart",
    "spain_laliga/2016-2017/2017-05-21 - 21-00 Barcelona 4 - 2 Eibar",
    "spain_laliga/2016-2017/2016-11-27 - 22-45 Real Sociedad 1 - 1 Barcelona",
    "europe_uefa-champions-league/2015-2016/2015-11-24 - 22-45 Barcelona 6 - 1 AS Roma",
    "spain_laliga/2016-2017/2016-10-02 - 17-15 Real Madrid 1 - 1 Eibar",
    "spain_laliga/2015-2016/2016-02-13 - 18-00 Real Madrid 4 - 2 Ath Bilbao",
    "italy_serie-a/2016-2017/2017-03-19 - 22-45 AS Roma 3 - 1 Sassuolo",
    "europe_uefa-champions-league/2016-2017/2017-02-15 - 22-45 Real Madrid 3 - 1 Napoli",
    "europe_uefa-champions-league/2015-2016/2015-09-15 - 21-45 Paris SG 2 - 0 Malmo FF",
    "spain_laliga/2015-2016/2015-12-20 - 18-00 Real Madrid 1 - 0 Rayo Vallecano",
    "spain_laliga/2016-2017/2016-10-29 - 21-45 Barcelona 1 - 0 Granada CF",
    "spain_laliga/2015-2016/2015-09-26 - 17-00 Barcelona 2 - 1 Las Palmas",

    # --- new batch ---
    "germany_bundesliga/2015-2016/2016-04-16 - 19-30 Bayern Munich 3 - 0 Schalke",
    "europe_uefa-champions-league/2015-2016/2015-09-29 - 21-45 Barcelona 2 - 1 Bayer Leverkusen",
    "spain_laliga/2016-2017/2017-01-14 - 18-15 Barcelona 5 - 0 Las Palmas",
    "spain_laliga/2016-2017/2017-03-12 - 22-45 Real Madrid 2 - 1 Betis",
    "italy_serie-a/2016-2017/2016-09-17 - 21-45 Napoli 3 - 1 Bologna",

    "england_epl/2015-2016/2016-05-07 - 17-00 Sunderland 3 - 2 Chelsea",
    "england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal",
    "england_epl/2015-2016/2015-08-08 - 19-30 Chelsea f2 - 2 Swansea",
    "england_epl/2015-2016/2015-10-24 - 17-00 West Ham 2 - 1 Chelsea",
    "england_epl/2016-2017/2016-11-06 - 17-15 Liverpool 6 - 1 Watford",
    "england_epl/2015-2016/2016-04-09 - 19-30 Manchester City 2 - 1 West Brom"
    "england_epl/2015-2016/2015-11-08 - 19-00 Arsenal 1 - 1 Tottenham",
    "england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester",

    "england_epl/2015-2016/2015-11-08 - 19-00 Arsenal 1 - 1 Tottenham",
    "england_epl/2016-2017/2016-10-22 - 17-00 Arsenal 0 - 0 Middlesbrough",
    "england_epl/2015-2016/2016-02-14 - 19-15 Manchester City 1 - 2 Tottenham",
    "england_epl/2015-2016/2016-03-20 - 19-00 Manchester City 0 - 1 Manchester United",
    "europe_uefa-champions-league/2014-2015/2015-03-18 - 22-45 Dortmund 0 - 3 Juventus",
    "europe_uefa-champions-league/2016-2017/2017-04-12 - 21-45 Bayern Munich 1 - 2 Real Madrid",
    "europe_uefa-champions-league/2016-2017/2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG",
    "europe_uefa-champions-league/2014-2015/2015-05-12 - 21-45 Bayern Munich 3 - 2 Barcelona",

]

for game in games:
    print(f"Downloading: {game}")
    downloader.downloadGame(
        game=game,
        files=["1_720p.mkv", "2_720p.mkv"]
    )
    print(f"Done: {game}\n")

#touch a marker so `make` knows new data is available and re-runs extraction (which is incremental — existing clips are skipped).
os.makedirs("data/soccernet", exist_ok=True)
open("data/soccernet/.download_stamp", "w").close()
print("Touched data/soccernet/.download_stamp — run `make` to process new games.")