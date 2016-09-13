#coding=utf-8
#coding=utf-8
import time
from splinter import Browser
import re
from time import sleep
import os
from splinter.browser import Browser
from selenium import webdriver
import selenium.webdriver.chrome.service as service


username=u'xlzhong123@163.com'
pwd=u'020002118'
url='http://www.mi.com/shouhuan2/'
count=0
#executable={'executable_path':'<C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe>'}
#chromedriver='C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe'
#b=Browser(driver_name='chrome') #模拟chrome浏览器登录
chrome= webdriver.Chrome('D:\Users\zhongxl\AppData\chromedriver.exe')



#b=Browser(driver_name='chrome') #模拟chrome浏览器登录
chrome.get(url)#需要打开的网页
chrome.find_element_by_name(u'登录').click()

b.find_by_value(u'立即登录').click()
sleep(1)
#print b.url
while b.url==url:
    try:
        if b.find_by_text(u'立即购买'):
            for i in b.find_by_text(u'立即购买'):
                i.click()
                sleep(0.1)
        else:
            print u'没找到'
    except:
        print u'暂时缺货'
    count+=1
    print u'正在尝试第%s次' %count
