import time


def scroll_to_bottom(driver):
    # 获取当前窗口的总高度
    js = 'return action=document.body.scrollHeight'
    # 初始化滚动条所在的高度
    height = 0
    # 当前窗口总高度
    currHeight = driver.execute_script(js)
    while True:
        # 将滚动条调整至页面底端
        for i in range(height, currHeight, 100):
            driver.execute_script("window.scrollTo(0, {})".format(i))
            time.sleep(0.02)
        height = currHeight
        time.sleep(2)
        currHeight = driver.execute_script(js)
        if height > 50000:
            break
        if height >= currHeight:
            break