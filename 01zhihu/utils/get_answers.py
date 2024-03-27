import time, json
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
    answerElementList = driver.find_elements(By.CSS_SELECTOR, "#QuestionAnswers-answers .List-item .ContentItem")
    print(answerElementList)
    return answerElementList, driver


def get_answers(answerElementList, url):
    data = pd.DataFrame(columns=('url', 'title', 'author_name', 'fans_count', 'created_time',
                                 'comment_count', 'voteup_count'))
    numAnswer = 0

    # 遍历每一个回答并获取回答中的信息
    for answer in answerElementList:
        dictText = json.loads(answer.get_attribute('data-zop'))
        question_title = dictText['title']
        author_name = dictText['authorName']
        if any(keyword in author_name for keyword in author_discard_keywords):
            continue
        created_time = answer.find_element(By.XPATH, "meta[@itemprop='dateCreated']").get_attribute(
            'content')  # 创建时间
        if int(created_time[:4]) < earliest_year:
            continue
        fans_count = answer.find_element(By.XPATH, "*//meta[contains(@itemprop, 'followerCount')]").get_attribute(
            'content')  # 粉丝数量
        comment_count = answer.find_element(By.XPATH, "meta[@itemprop='commentCount']").get_attribute(
            'content')  # 评论数量
        voteup_count = answer.find_element(By.XPATH, "meta[@itemprop='upvoteCount']").get_attribute(
            'content')  # 赞同数量
        contents = answer.find_elements(By.TAG_NAME, "p")
        content = '\n'.join([content.text for content in contents])
        time.sleep(0.001)

        if len(content) < 50:
            continue

        row = {'url': [url],
               'title': [question_title],
               'author_name': [author_name],
               'fans_count': [fans_count],
               'created_time': [created_time],
               'comment_count': [comment_count],
               'voteup_count': [voteup_count],
               'content': [content]
               }
        data = data.append(pd.DataFrame(row), ignore_index=True)
        numAnswer += 1
        print(f"问题：【{question_title}】 的第 {numAnswer} 个回答抓取完成...")
        time.sleep(0.2)

    return data, question_title


class GetAnswers():
    def __init__(self, urls):
        self.urls = urls  # 传过来的是questions_url

    def main(self):
        data = pd.DataFrame(columns=('url', 'title', 'author_name', 'fans_count', 'created_time',
                                     'comment_count', 'voteup_count'))

        print('需要抓取的问题数量：', len(self.urls))

        for url in tqdm(self.urls, desc='获取问答中'):
            print('----------------------------------------')
            print('question_url: ', url)

            try:
                time.sleep(random.uniform(1, 3))
                answerElementList, driver = get_html(url)

                print("开始抓取该问题的回答...")
                answerData, question_title = get_answers(answerElementList, url)
                if answerData.empty:
                    print(f"问题【{question_title}】 没有{earliest_year}年之后的回答...")
                    continue
                driver.close()
                print(f"问题：【{question_title}】 的回答全部抓取完成...")
                time.sleep(random.uniform(1, 3))

                data = data.append(answerData, ignore_index=True)

            except Exception as e:
                print(e)
                print(f"[ERROR] 抓取失败...")
                time.sleep(random.uniform(2, 4))
                continue
        return data


if __name__ == '__main__':
    urls = ['https://www.zhihu.com/question/638174866', 'https://www.zhihu.com/question/638174866']
    data = GetAnswers(urls).main()
    print(1)

