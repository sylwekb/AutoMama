import getpass
import os

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

options = webdriver.ChromeOptions()

profile = {
    "plugins.plugins_list": [
        {"enabled": False, "name": "Chrome PDF Viewer"}
    ],  # Disable Chrome's PDF Viewer
    "download.default_directory": os.path.expanduser("~/Downloads"),
    "download.extensions_to_open": "applications/pdf",
}

options.add_experimental_option("prefs", profile)
driver = webdriver.Chrome('chromedriver', chrome_options=options)  # Optional argument, if not specified will search path.

driver.get("https://www.domeny.tv/admin/index.php?page=92")
elem = driver.find_element_by_id("username")
elem.clear()
elem.send_keys("emandero")

elem = driver.find_element_by_name("form_password")
elem.clear()
elem.send_keys(getpass.getpass())


elem.send_keys(Keys.RETURN)

driver.get("https://www.domeny.tv/admin/index.php?page=92")
pdfs = driver.find_elements_by_xpath("//a/img")


for element in pdfs:
    element.click()

driver.close()
