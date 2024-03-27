import time, json, re
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm

from utils.config import driver_path, earliest_year, author_discard_keywords
from utils.driver_utils import scroll_to_bottom


def get_html(url):
    driver = webdriver.Chrome(executable_path=driver_path)
    driver.implicitly_wait(10)
    driver.maximize_window()
    driver.get(url)
    time.sleep(random.uniform(1, 2))

    close_btn = driver.find_element(By.XPATH, "//button[@class='Button Modal-closeButton Button--plain']")  # 定位登录界面关闭按钮
    close_btn.click()

    scroll_to_bottom(driver)
    mainElement = driver.find_elements(By.CSS_SELECTOR, ".Post-content")[0]
    print(mainElement)
    return mainElement, driver


def get_fans(user_link):
    driver = webdriver.Chrome(executable_path=driver_path)
    driver.implicitly_wait(10)
    driver.maximize_window()
    driver.get(user_link)
    time.sleep(random.uniform(1, 2))

    close_btn = driver.find_element(By.XPATH, "//button[@class='Button Modal-closeButton Button--plain']")  # 定位登录界面关闭按钮
    close_btn.click()

    fans_count = driver.find_elements(By.CSS_SELECTOR, ".NumberBoard-itemValue")[1].text  # 定位登录界面关闭按钮
    driver.close()
    return fans_count  # 有可能是‘6.7 万’


def get_passage(mainElement, url, driver):
    data = pd.DataFrame(columns=('url', 'title', 'author_name', 'fans_count', 'created_time',
                                 'comment_count', 'voteup_count'))

    dictText = json.loads(mainElement.get_attribute('data-zop'))
    passage_title = dictText['title']
    author_name = dictText['authorName']
    if any(keyword in author_name for keyword in author_discard_keywords):
        return None, None

    # 粉丝数量
    user_link = mainElement.find_element(By.CSS_SELECTOR, ".UserLink-link").get_attribute('href')  # 定位登录界面关闭按钮
    fans_count = get_fans(user_link)

    # 创建时间
    created_time = mainElement.find_element(By.CSS_SELECTOR, ".ContentItem-time").text
    created_time = re.search('\d{4}-\d{2}-\d{2} \d{2}:\d{2}', created_time).group()
    if int(created_time[:4]) < earliest_year:
        return None, None


    # 评论数量
    comment_count = mainElement.find_element(By.CSS_SELECTOR, ".BottomActions-CommentBtn").text
    comment_count = re.search('\d+', comment_count)
    if comment_count:
        comment_count = comment_count.group(0)
    else:
        comment_count = 0

    voteup_count = mainElement.find_element(By.CSS_SELECTOR, ".css-1lr85n").text  # 赞同数量
    voteup_count = re.search('\d+', voteup_count)
    if voteup_count:
        voteup_count = voteup_count.group(0)
    else:
        voteup_count = 0

    # 文章内容
    contents = mainElement.find_element(By.CSS_SELECTOR, ".RichText").find_elements(By.TAG_NAME, "p")
    content = '\n'.join([content.text for content in contents])
    time.sleep(0.001)

    row = {'url': [url],
           'title': [passage_title],
           'author_name': [author_name],
           'fans_count': [fans_count],
           'created_time': [created_time],
           'comment_count': [comment_count],
           'voteup_count': [voteup_count],
           'content': [content]
           }
    data = data.append(pd.DataFrame(row), ignore_index=True)  # 看一下
    return data, passage_title


class GetPassages():
    def __init__(self, urls):
        self.urls = urls  # 传过来的是questions_url

    def main(self):
        data = pd.DataFrame(columns=('url', 'title', 'author_name', 'fans_count', 'created_time',
                                     'comment_count', 'voteup_count', 'content'))

        print('需要抓取的文章数量：', len(self.urls))

        for url in tqdm(self.urls, desc='获取文章中'):
            print('----------------------------------------')
            print('passage_url: ', url)

            try:
                time.sleep(random.uniform(1, 3))
                mainElement, driver = get_html(url)

                # print("开始抓取该文章...")
                passageData, passage_title = get_passage(mainElement, url, driver)
                if passageData.empty:
                    print(f"【{passage_title}】 不是{earliest_year}年之后的文章...")
                    continue
                driver.close()
                print(f"文章：【{passage_title}】 抓取完成...")
                time.sleep(random.uniform(1, 3))

                data = data.append(passageData, ignore_index=True)

            except Exception as e:
                print(e)
                print(f"[ERROR] 抓取失败...")
                time.sleep(random.uniform(2, 4))
                continue
        return data


if __name__ == '__main__':
    urls = ['https://zhuanlan.zhihu.com/p/666940612', 'https://zhuanlan.zhihu.com/p/666940612']
    data = GetPassages(urls).main()
    print(1)

