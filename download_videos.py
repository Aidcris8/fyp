from SoccerNet.Downloader import SoccerNetDownloader

PASSWORD = "s0cc3rn3t"

downloader = SoccerNetDownloader(LocalDirectory="data/soccernet")
downloader.password = PASSWORD

games = [
    "england_epl/2016-2017/2016-10-02 - 18-30 Burnley 0 - 1 Arsenal",
    "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea",
    "england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool",
    "england_epl/2015-2016/2015-11-29 - 15-00 Tottenham 0 - 0 Chelsea",


]

for game in games:
    print(f"Downloading: {game}")
    downloader.downloadGame(
        game=game,
        files=["1_720p.mkv", "2_720p.mkv"]
    )
    print(f"Done: {game}\n")