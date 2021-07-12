from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import warnings
import zipfile
import shutil
import numpy
import time
import glob
import re
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Crawling code
def crawling():
    # Execute webdriver
    wd = webdriver.Chrome(ChromeDriverManager().install())  # webdriver path
    wd.get('https://fonts.google.com/?category=Serif,Sans+Serif,Display&subset=latin')
    wd.maximize_window()

    font_list = []
    for i in range(150):
        if len(font_list) < 1000:
            last_font = []
            win_height = 925 * i  # height of chrome window
            scroll = "window.scrollTo(1920, " + str(win_height) + ");"
            wd.execute_script(scroll)
            html = wd.page_source
            font_name = re.findall("(-6\"\>+[0-9a-zA-Z\s]+\<)", html)
            for font in font_name:
                style = font[4:-1]
                if style not in font_list:
                    font_list.append(style)
                    print(style)
                else:
                    print("overlap!")
                    pass

            for font in font_list:
                if ' SC' in str(font):
                    font_list.remove(font)
                    print("remove_small_caps:", font)
                else:
                    pass

            for font in font_list:
                if 'Barcode' in str(font):
                    font_list.remove(font)
                    print("remove_barcode", font)
                else:
                    pass

            print(len(font_list))
            time.sleep(1.5)

        else:
            break

    wd.close()

    if 'Kumar One' in font_list:
        font_list.remove('Kumar One')
    if 'Kumar One Outline' in font_list:
        font_list.remove('Kumar One Outline')

    # font_list = font_list[:100]  # 원하는 폰트 수
    font_df = pd.DataFrame(font_list)
    font_df.columns = {"font_name"}

    return font_df.to_csv("./font_list.csv", encoding='utf-8', mode='w', index=False)


def remove_overlap():
    df = pd.read_csv("./font_list.csv")
    font_list = df['font_name'].to_list()

    overlap_list = []
    for i in font_list:
        font_list_compared = df['font_name'].to_list()
        font_list_compared.remove(i)
        for j in font_list_compared:
            if str(i) in str(j):
                overlap_list.append(j)
            else:
                pass

    font_list = [x for x in font_list if x not in overlap_list]
    print(len(font_list))
    font_df = pd.DataFrame(font_list)
    font_df.columns = {"font_name"}

    return font_df.to_csv("./font_list.csv", encoding='utf-8', mode='w', index=False)


# Download ttf files
def download_ttfs():
    # load font list
    font_list_df = pd.read_csv('./font_list.csv')
    font_list = list(font_list_df.font_name)
    style_name = [i.replace(" ", "+") for i in font_list]
    style_name.sort()

    # execute webdriver
    os.makedirs("./ttf_zips", exist_ok=True)
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {
        "download.default_directory": os.path.abspath('./ttf_zips')})
    wd = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
    wd.get("https://fonts.google.com/")
    wd.maximize_window()

    # ttf file crawling by zip file
    for style in style_name:
        wd.get("https://fonts.google.com/specimen/" + style)
        WebDriverWait(wd, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                    "#main-content > gf-sticky-header > div > div > "
                                                                    "button > span.mat-button-wrapper > span")))
        button = wd.find_element_by_css_selector(
            "#main-content > gf-sticky-header > div > div > button > span.mat-button-wrapper > span")
        wd.implicitly_wait(60)  # setting waiting time for downloading bigsize files
        button.click()


# Unzip
def unzip():
    # load font list
    zip_list = os.listdir("./ttf_zips")
    print(zip_list)
    os.makedirs("./ttfs", exist_ok=True)

    # decompressing zip files
    for zip in zip_list:
        ttf_zip = zipfile.ZipFile("./ttf_zips/" + zip)
        for file in ttf_zip.namelist():
            ttf_zip.extract(file, "./ttfs/")
        ttf_zip.close()


# Collect list of all ttf files
def read_all_file(path):
    output = os.listdir(path)
    file_list = []

    for i in output:
        if os.path.isdir(path + "/" + i):
            file_list.extend(read_all_file(path + "/" + i))
        elif os.path.isfile(path + "/" + i):
            file_list.append(path + "/" + i)

    return file_list


# Copy files in new path
def copy_all_file(file_list, new_path):
    for src_path in file_list:
        file = src_path.split("/")[-1]
        shutil.copyfile(src_path, new_path + "/" + file)


# Execute arranging
def arrange():
    if os.path.isdir("./ttfs/static"):
        file_list = read_all_file("./ttfs/static")
        copy_all_file(file_list, "./ttfs")
        shutil.rmtree("./ttfs/static")
        os.chdir("./ttfs")
        otfs = glob.glob("*.otf")
        txts = glob.glob("*.txt")
        for_remove = otfs + txts
        for file in for_remove:
            if not os.path.exists("./" + file):
                pass
            else:
                os.remove("./" + file)
    else:
        pass


def get_regular():
    ttf_list = os.listdir('./ttfs')
    for ttf in ttf_list:
        if 'Regular' not in str(ttf):
            os.remove("./ttfs/" + ttf)
        elif 'Semi' in str(ttf):
            os.remove("./ttfs/" + ttf)
        else:
            continue

    return print("finished filtering")
