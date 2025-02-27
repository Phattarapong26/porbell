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
    chrome_options.add_argument("--window-size=1920,1080")
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

def get_total_pages(driver):
    try:
        pagination = driver.find_element(By.CLASS_NAME, "pagination")
        pages = pagination.find_elements(By.TAG_NAME, "li")
        if pages:
            last_page = pages[-2].text  # Last page number is usually second to last element
            return int(last_page)
    except:
        return 1
    return 1

def save_progress(data, filename='property_listings_progress.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    base_url = "https://www.livinginsider.com/searchword/Condo/all/1/รวมประกาศขาย-เช่าคอนโด.html"
    driver = setup_driver()
    all_data = []
    
    try:
        # Get first page to determine total pages
        driver.get(base_url.format(1))
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "istock-list")))
        
        total_pages = get_total_pages(driver)
        print(f"Total pages to scrape: {total_pages}")
        
        # Loop through all pages
        for page in range(1, total_pages + 1):
            print(f"Processing page {page}/{total_pages}")
            
            # Load page
            if page > 1:
                driver.get(base_url.format(page))
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "istock-list")))
            
            # Get all listings on current page
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            listings = soup.find_all('div', class_='istock-list')
            
            # Process each listing
            for idx, listing in enumerate(listings, 1):
                try:
                    listing_id = listing.get('id', '').replace('list', '')
                    print(f"Processing listing {idx} on page {page} (ID: {listing_id})")
                    
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
                        
                        # Save progress after each listing
                        save_progress(all_data)
                        
                        time.sleep(1)  # Be nice to the server
                except Exception as e:
                    print(f"Error processing listing {listing_id}: {str(e)}")
                    continue
            
            # Save progress after each page
            save_progress(all_data)
            
        # Final save to main file
        with open('property_listings_final.json', 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
            
        print(f"Completed! Total listings scraped: {len(all_data)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Save whatever data we have
        save_progress(all_data, 'property_listings_error.json')
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
