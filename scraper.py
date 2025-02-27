from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu") 
    chrome_options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def extract_listing_data(soup, listing_id):
    data = {
        'listing_id': listing_id,
        'basic_info': {},
        'details': {},
        'location': {},
        'images': [],
        'facilities': [],
        'contact': {}
    }
    
    # Basic Information
    title = soup.find('h1', class_='font_sarabun')
    if title:
        data['basic_info']['title'] = title.text.strip()
    
    price_div = soup.find('div', class_='listing-cost')
    if price_div:
        original_price = price_div.find('div', class_='txt-before-disc')
        current_price = price_div.find('div', class_='t-16')
        data['basic_info']['original_price'] = original_price.text.strip() if original_price else None
        data['basic_info']['current_price'] = current_price.text.strip() if current_price else None

    # Property Details
    property_details = soup.find_all('div', class_='ic-detail')
    for detail in property_details:
        text = detail.text.strip()
        if 'ตร.ว.' in text:
            data['details']['area'] = text
        elif 'ชั้น' in text:
            data['details']['floors'] = text
        elif 'ห้องนอน' in text:
            data['details']['bedrooms'] = text
        elif 'ห้องน้ำ' in text:
            data['details']['bathrooms'] = text

    # Location Information
    location_elem = soup.find('div', class_='ic-detail-zone')
    if location_elem:
        data['location']['area'] = location_elem.text.strip()
        
    project_name = soup.find('div', class_='detail-text-project')
    if project_name:
        data['location']['project'] = project_name.text.strip()

    # Images
    gallery = soup.find('div', id='animated-thumbnails')
    if gallery:
        for img in gallery.find_all('a', class_='gallery-item'):
            if img.get('data-src'):
                data['images'].append(img['data-src'])

    # Facilities
    facilities = soup.find_all('div', class_='col-xs-6 col-sm-6 col-md-6 d-flex mb-2')
    for facility in facilities:
        if facility.find('span', class_='text-additional'):
            data['facilities'].append(facility.find('span', class_='text-additional').text.strip())

    # Contact Information
    contact_box = soup.find('div', class_='box_group_chat')
    if contact_box:
        line_elem = contact_box.find('a', class_='co-line')
        if line_elem:
            data['contact']['line'] = line_elem.get('data-url')
            
        email_elem = contact_box.find('a', class_='co-email')
        if email_elem:
            data['contact']['email'] = True

    # Description
    desc_elem = soup.find('div', class_='wordwrap')
    if desc_elem:
        data['details']['description'] = desc_elem.text.strip()

    return data

def main():
    url = "https://www.livinginsider.com/searchword/Townhouse/all/1/รวมประกาศขาย-เช่า-ทาวน์เฮ้าส์-ทาวน์โฮม-ทุกทำเล.html"
    driver = setup_driver()
    
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "istock-list")))
        
        # Get all listings
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        listings = soup.find_all('div', class_='istock-list')
        
        all_data = []
        
        # Process each listing
        for listing in listings[:5]: # Limit to 5 listings for testing
            listing_id = listing.get('id', '').replace('list', '')
            
            # Get detail page URL
            detail_link = listing.find('a', class_='image-ratio-4-3')
            if detail_link and detail_link.get('href'):
                # Visit detail page
                driver.get(detail_link['href'])
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "detail-content")))
                
                # Extract detailed information
                detail_soup = BeautifulSoup(driver.page_source, 'html.parser')
                listing_data = extract_listing_data(detail_soup, listing_id)
                all_data.append(listing_data)
                
                time.sleep(1) # Be nice to the server
        
        # Save to JSON file
        with open('property_listings.json', 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
            
        print("Data has been saved to property_listings.json")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
