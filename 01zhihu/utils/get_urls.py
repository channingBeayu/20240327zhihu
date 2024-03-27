import time, json, re
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from utils.config import driver_path
from utils.driver_utils import scroll_to_bottom


def get_html(search_key):
    # 知乎有反爬机制，参照https://blog.csdn.net/zhuan_long/article/details/109800202
    # 事先打开一个浏览器，然后让selenium接管即可
    # 【命令行】chrome.exe --remote-debugging-port=9222 --user-data-dir="D:\tmp"
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")  # 前面设置的端口号
    driver = webdriver.Chrome(executable_path=driver_path, options=chrome_options)
    driver.implicitly_wait(10)
    driver.maximize_window()

    driver.get('https://www.zhihu.com/search?type=content&q=' + search_key)
    time.sleep(random.uniform(1, 2))

    # 搜索
    driver.find_element(By.CSS_SELECTOR, "#Popover1-toggle").send_keys('哈尔滨旅游感受')
    driver.find_element(By.CSS_SELECTOR, ".SearchBar-searchButton").click()
    time.sleep(random.uniform(1, 2))

    scroll_to_bottom(driver)

    tags_a = driver.find_elements(By.XPATH, "//a[@data-za-detail-view-id='3942']")
    urls_question = set()
    urls_passage = set()
    for tag_a in tags_a:
        url = tag_a.get_attribute('href')
        if 'question' in url:
            urls_question.add(url.split('/answer')[0])
        else:
            urls_passage.add(url)

    return urls_question, urls_passage




if __name__ == '__main__':
    urls_question, urls_passage = get_html('哈尔滨旅游感受')
    print(1)

