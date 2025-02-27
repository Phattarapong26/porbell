from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
# เพิ่มบรรทัดนี้
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

# ... ส่วนที่เหลือของโค้ดเหมือนเดิม ...


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def extract_listing_data(listing):
    data = {}
    
    # Extract listing ID
    data['listing_id'] = listing.get('id', '').replace('list', '')
    
    # Extract price information
    price_div = listing.find('div', class_='listing-cost')
    if price_div:
        data['original_price'] = price_div.find('div', class_='txt-before-disc').text.strip() if price_div.find('div', class_='txt-before-disc') else ''
        data['current_price'] = price_div.find('div', class_='t-16').text.strip() if price_div.find('div', class_='t-16') else ''
    
    # Extract title
    title_elem = listing.find('p', class_='font-Sarabun')
    data['title'] = title_elem.text.strip() if title_elem else ''
    
    # Extract location
    location_elem = listing.find('div', class_='ic-detail-zone')
    data['location'] = location_elem.text.strip() if location_elem else ''
    
    # Extract property details
    details = listing.find_all('div', class_='ic-detail')
    for detail in details:
        text = detail.text.strip()
        if 'ตร.ว.' in text:
            data['area'] = text
        elif 'ชั้น' in text:
            data['floors'] = text
        elif 'ห้องนอน' in text:
            data['bedrooms'] = text
        elif 'ห้องน้ำ' in text:
            data['bathrooms'] = text
    
    # Extract last update and views
    date_view = listing.find('div', class_='crad-date-view')
    if date_view:
        data['last_updated'] = date_view.find('div', class_='istock-lastdate').text.strip() if date_view.find('div', class_='istock-lastdate') else ''
        data['views'] = date_view.find('div', class_='istock-view').text.strip() if date_view.find('div', class_='istock-view') else ''
    
    # Extract image URL
    img_elem = listing.find('img', class_='img-responsive')
    data['image_url'] = img_elem['src'] if img_elem else ''
    
    return data

def main():
    url = "https://www.livinginsider.com/searchword/Townhouse/all/1/รวมประกาศขาย-เช่า-ทาวน์เฮ้าส์-ทาวน์โฮม-ทุกทำเล.html"
    driver = setup_driver()
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "istock-list")))
        
        # Get page source and create BeautifulSoup object
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        listings = soup.find_all('div', class_='istock-list')
        
        # Extract data from all listings
        all_data = []
        for listing in listings:
            listing_data = extract_listing_data(listing)
            all_data.append(listing_data)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_data)
        
        # Reorder columns for better readability
        columns_order = [
            'listing_id', 'title', 'current_price', 'original_price',
            'location', 'area', 'floors', 'bedrooms', 'bathrooms',
            'last_updated', 'views', 'image_url'
        ]
        df = df[columns_order]
        
        # Save to CSV with Thai character support
        df.to_csv('property_listings.csv', index=False, encoding='utf-8-sig')
        print("Data has been saved to property_listings.csv")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
