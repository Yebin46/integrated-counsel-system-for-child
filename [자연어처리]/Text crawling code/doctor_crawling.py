# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:09:15 2018
@author: jingyu
refer: "https://github.com/jingyu12/healthcare_counseling_service/blob/master/crawling_code/doctor_scraping.py" 
"""

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
from selenium.webdriver.common.keys import Keys

import os
import pandas as pd
import numpy as np
from itertools import chain
import time

# os.chdir('D:/myworks/18-1/ssem')

# get answer_count
def scraping_at_one_page(driver,cur_page):
    title_list=[]
    question_list=[]
    answer_list=[]
    category_list=[]
    for i in range(1,21):        
        elem=driver.find_element_by_xpath('//*[@id="au_board_list"]/tr['+str(i)+']/td[1]/a')
        elem.click()
        # alarm check
        try:
            driver.switch_to_alert().accept()
        except:
            pass
        # try:            
        # title=driver.find_element_by_xpath("//div[@class='c-heading__title']/div/div").text 
        try: 
            title = driver.find_element_by_class_name("title").text
            print(title)
            print('\n')
            if len(title) > 150:
                pass
        except:
            title = "no title"
            print('no title')

        # question=driver.find_element_by_xpath("//div[@class='c-heading__content']").text
        try:
            question = driver.find_element_by_class_name('c-heading__content').text
            print(question)
            print('\n')
        except:
            question = "no question"
            print('no question')
            if len(question) > 150:
                pass
        
        # question = driver.find_element_by_class_name('c-heading__content').text
        try:
            answer=driver.find_element_by_xpath("//div[@class='_endContentsText c-heading-answer__content-user']").text
            print(answer)
            print('\n')
        except:
            answer = "no answer"
            print('no answer')

        title_list.append(title)
        question_list.append(question)
        answer_list.append(answer)
        driver.execute_script("window.history.go(-1)")
        
        try:
            category=driver.find_element_by_xpath("//*[@id='au_board_list']/tr['+str(i)+']/td[2]/a").text
            print(category)
            print('\n')
        except:
            category = "no category"
            print("no category")
            
        category_list.append(category)
        time.sleep(0.7)
        
        # except NoSuchElementException:
            # exit()
            # driver.execute_script("window.history.go(-1)")
        
    print('{} Q&A iteration is done'.format(cur_page))
    time.sleep(5)

    return category_list, title_list,question_list,answer_list
    

# scraping whole page
def scraping_full_page(page_number,doctor_num):
    c_l=[]
    t_l=[]
    q_l=[]
    a_l=[]

    for i in range(2,page_number):
        # try:
        category,title,question,answer=scraping_at_one_page(driver,i)
        c_l.extend(category)
        t_l.extend(title)
        q_l.extend(question)
        a_l.extend(answer)

        new_url = doctor_home_list[int(doctor_num)]+'&page='+str(i)
        driver.get(new_url)

    
    print('One doctor iteration is done')
    print(len(c_l))
    print(len(t_l))
    print(len(q_l))
    print(len(a_l))

    
    df=pd.DataFrame({'category':c_l,"title":t_l,"question":q_l, "answer":a_l})
    
    return df

    # return c_l,t_l,q_l,a_l



##################
# start scraping
##################  
path = "/Users/hyebin/opt/anaconda3/envs/study/chromedriver"
driver=webdriver.Chrome(path)
doctor_home_list=[#'https://kin.naver.com/userinfo/expert/answerList.nhn?u=QbPYSQHbEVHsy13frYO7Wkq9LJla1qotlZkVSwYDsqk%3D'] # 권순모 - 일반 정신질환, 우울증. 공황장애. 불안장애. 수면장애, 알콜 등
                # 'https://kin.naver.com/userinfo/expert/answerList.nhn?u=HaNfIgmblpoaWZLW%2FUqzLu7EcB%2BNPc9ybXr1CIjefDI%3D'] # 배성범 - 심층심리상담
                # 'https://kin.naver.com/userinfo/expert/answerList.nhn?u=c3IUZFm5kpUWmV94ptcMz8UaIIwRRlRZWhqbxzsYPzE%3D'# 최성환 - 소아청소년 정신과,치매, 노인정신 기억장애
                # 'https://kin.naver.com/userinfo/expert/answerList.nhn?u=tYuqffLqOiT3XXFR7EcoqF5jEJqX8VYYEvjH6cUYo1U%3D' # 김봉수 - 소아 정신건강의학과
                # 'https://kin.naver.com/userinfo/expert/answerList.nhn?u=m6V%2BptYKBpGQrHfBYvzaxAACCmWzKqpDhHsUBMZ3Opw%3D' # 김슬기 - 조현병,식이장애, 우울증, 불안장애
                'https://kin.naver.com/userinfo/expert/answerList.nhn?u=yaW6qySX89rRGjhXvapk6Kx35eidKtA9aBaVTOUjGlE%3D' # 황선희 - 일반 정신질환,소아청소년 정신과
                ]
                  


def run_scraping():    
    full_df=pd.DataFrame()
    for index,url in enumerate(doctor_home_list):
        driver.get(url)
        # answer_count=driver.find_element_by_xpath('//*[@id="content"]/dl/dd[1]').text
        answer_count = driver.find_element_by_xpath("//dl[@class='my_spec']/dd[1]").text
        answer_count = "".join(answer_count.split(','))
        # page_number=int(np.round(int(answer_count)/20))
        page_number = 30
        print('The page number of this doctor is :',page_number)    
        df=scraping_full_page(page_number, index)
        
        full_df=full_df.append(df)
    
    return full_df

df=run_scraping()

pd.DataFrame.to_csv(df,'result/health_care_qa_황선희.csv')
driver.quit()