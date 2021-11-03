import urllib
from pathlib import Path
from urllib import request
import io
import pandas as pd
import requests
import zipfile

DATA_DIR = Path.cwd() / "data"

########################################################################################

url_root = (
    "https://data.stadt-zuerich.ch/dataset/"
    + "3d0c33d6-ec57-426a-918c-ac8a60573789/resource/"
)

url_2021 = (
    url_root
    + "75ace201-1275-4691-b196-69c19f4d9e2a/download/"
    + "ugz_ogd_meteo_d1_2021.csv"
)
url_2020 = (
    url_root
    + "7d1daeea-0276-4b68-bb4a-5dd47cac2206/download/"
    + "ugz_ogd_meteo_h1_2020.csv"
)
url_2019 = (
    url_root
    + "56d5475a-fbed-4f42-a317-47fc76ab893b/download/"
    + "ugz_ogd_meteo_h1_2019.csv"
)
url_2018 = (
    url_root
    + "78b70fc7-5df1-4d37-981e-8eceb5553c38/download/"
    + "ugz_ogd_meteo_h1_2018.csv"
)
url_2017 = (
    url_root
    + "291bc977-8809-4b8e-94ae-74a5b3acff93/download/"
    + "ugz_ogd_meteo_h1_2017.csv"
)
url_2016 = (
    url_root
    + "957ca66c-e297-4cb2-b807-0c4671a3a3f0/download/"
    + "ugz_ogd_meteo_h1_2016.csv"
)
url_2015 = (
    url_root
    + "1fab3154-8149-4e31-8673-a5d64f7b37aa/download/"
    + "ugz_ogd_meteo_h1_2015.csv"
)
url_2014 = (
    url_root
    + "119f6798-7aea-4117-b91e-ac6756558637/download/"
    + "ugz_ogd_meteo_h1_2014.csv"
)
url_2013 = (
    url_root
    + "f508fe0a-68e8-453a-81a8-5439fbfe94fd/download/"
    + "ugz_ogd_meteo_h1_2013.csv"
)
url_2012 = (
    url_root
    + "c02bc5d3-bbf6-4e38-9e12-87699878d6a5/download/"
    + "ugz_ogd_meteo_h1_2012.csv"
)
url_2011 = (
    url_root
    + "17bd93c7-4e67-4e11-b9d2-59e6fd77a87b/download/"
    + "ugz_ogd_meteo_h1_2011.csv"
)
url_2010 = (
    url_root
    + "a67459d8-4822-4f22-ae45-df1693012525/download/"
    + "ugz_ogd_meteo_h1_2010.csv"
)

URL_LIST = [
    url_2010,
    url_2011,
    url_2012,
    url_2013,
    url_2014,
    url_2015,
    url_2016,
    url_2017,
    url_2018,
    url_2019,
    url_2020,
    url_2021,
]


########################################################################################
def fetch_meteo_zh_data() -> None:
    ugz_ogd_meteo_all = pd.DataFrame()
    for url in URL_LIST:
        container_df = pd.read_csv(url)
        ugz_ogd_meteo_all = pd.concat([ugz_ogd_meteo_all, container_df])

    file_dir = str(DATA_DIR / "ugz_ogd_meteo_all.feather")
    ugz_ogd_meteo_all.astype({"Datum": "datetime64[ns]"}).reset_index().drop(
        columns=["index"]
    ).to_feather(file_dir)
    print(file_dir)


########################################################################################
def _construct_bag_download_links() -> list:
    url_root_bag = "https://www.covid19.admin.ch"
    url_scrap = url_root_bag + "/en/hosp-capacity/icu/d/development?time=total"
    url_request_open = urllib.request.urlopen(url_scrap)
    url_source_txt = str(url_request_open.read())
    url_request_open.close()
    url_source_split = url_source_txt.split('"')
    list_url_download_zip = [
        url_root_bag + x
        for x in url_source_split
        if x.startswith("/api") & x.endswith(".zip")
    ]
    return list_url_download_zip


def _download_from_url_unzip(zip_file_url: str, save_path: Path):
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(save_path)


def fetch_bag_data() -> None:
    list_zips = _construct_bag_download_links()
    for zip_file in list_zips:
        _download_from_url_unzip(zip_file_url=zip_file, save_path=DATA_DIR / "bag")

    hospital_file = DATA_DIR / "bag" / "data" / "COVID19HospCapacity_geoRegion.csv"
    hospital_df = (
        pd.read_csv(hospital_file)
        .astype({"date": "datetime64[ns]"})
        .sort_values("date")
        .reset_index()
        .drop(columns="index")
    )
    file_dir = DATA_DIR / "hospital_capacity.feather"
    hospital_df.to_feather(file_dir)
    print(file_dir)
    print(f"Last date: {hospital_df.date.max()}")


########################################################################################
# Execute
########################################################################################
fetch_meteo_zh_data()


fetch_bag_data()
