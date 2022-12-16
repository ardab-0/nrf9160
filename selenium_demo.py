from pprint import pprint

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.firefox.options import Options





class WebCrawler:
    def __init__(self):
        options = Options()
        options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'
        self.driver = webdriver.Firefox(options=options)
        self.driver.get("http://www.cell2gps.com/")

    def __del__(self):
        self.driver.close()

    def get_location_from_page(self, form_details):
        form_elem = self.driver.find_element(By.ID, "form1")
        form_parts = form_elem.find_elements(By.TAG_NAME, "input")

        i = 0
        while i < 4:
            form_parts[i].clear()
            form_parts[i].send_keys(str(form_details[i]))
            i += 1

        myElement = self.driver.find_element(By.TAG_NAME, "button")
        webdriver.ActionChains(self.driver).move_to_element(myElement).click(myElement).perform()
        time.sleep(5)
        page_source = self.driver.page_source
        start_idx = page_source.find("location is")
        end_idx = page_source.find("Accuracy", start_idx)
        position = page_source[start_idx + 13: end_idx - 2]
        position = position.split(",")
        position_tuple = (float(position[0]) , float(position[1]))
        return position_tuple


crawler = WebCrawler()
form_details = [262, 1, 22549, 44414465]
print(crawler.get_location_from_page(form_details))

form_details = [262, 1, 22549, 46791175]
print(crawler.get_location_from_page(form_details))