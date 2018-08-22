from pprint import pprint

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Chrome()
driver.get("https://www.gumtree.pl/s-wypozyczalnie/poznan/kamper/v1c9215l3200366q0p1")

links = [e.get_attribute('href') for e in driver.find_elements_by_css_selector("a.href-link")]

ret = {}
for link in links:
    driver.get(link)
    print(f"working on {link}")
    title = driver.find_element_by_css_selector("span.myAdTitle").text

    description = driver.find_element_by_css_selector("div.vip-details div.description")

    # click to reveal phone number
    try:
        driver.find_element_by_css_selector("a.tmpNumber").click()
    except NoSuchElementException:
        phone_number = None
    else:
        phone_number = driver.find_element_by_css_selector("div.vip a.button span.label").text

    ret[link] = {
        'title': title,
        'description': description,
        'phone_number': phone_number,
    }

pprint(ret)
