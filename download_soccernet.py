from SoccerNet.Downloader import SoccerNetDownloader

PASSWORD = "s0cc3rn3t"

downloader = SoccerNetDownloader(LocalDirectory="data/soccernet")
downloader.password = PASSWORD

downloader.downloadGames(
    files=["Labels-v2.json"],
    split=["train", "valid", "test"]
)