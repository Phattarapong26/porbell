from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os
import re
import json

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def extract_listing_card_data(listing):
    """Extract comprehensive data from the listing card on the search results page"""
    card_data = {
        'listing_id': listing.get('id', '').replace('list', ''),
        'title': None,
        'price': None,
        'area': None,
        'floor': None,
        'bedrooms': None,
        'bathrooms': None,
        'location': None,
        'project': None,
        'listing_type': None,
        'property_type': None,
        'condition': None,
        'detail_url': None,
        'image_url': None,
        'last_updated': None
    }
    
    # Extract title
    title_elem = listing.find('p', class_='font-Sarabun')
    if title_elem:
        card_data['title'] = title_elem.text.strip()
    
    # Extract price
    price_elem = listing.find('div', class_='text_price')
    if price_elem:
        # Remove the ฿ symbol and any commas
        price_text = price_elem.text.strip()
        card_data['price'] = price_text
    
    # Extract location/project
    location_elem = listing.find('div', class_='ic-detail-zone')
    if location_elem:
        card_data['location'] = location_elem.text.strip()
        # Often the location element contains the project name
        card_data['project'] = location_elem.text.strip()
    
    # Extract property details
    property_details = listing.find_all('div', class_='ic-detail')
    for detail in property_details:
        text = detail.text.strip()
        if 'ตร.ม.' in text or 'ตร.ว.' in text:
            card_data['area'] = text
        elif 'ชั้น' in text:
            # Extract just the floor number
            floor_match = re.search(r'(\d+)', text)
            if floor_match:
                card_data['floor'] = floor_match.group(1)
            else:
                card_data['floor'] = text
        elif 'ห้องนอน' in text:
            # Extract just the number
            bedroom_match = re.search(r'(\d+)', text)
            if bedroom_match:
                card_data['bedrooms'] = bedroom_match.group(1)
            else:
                card_data['bedrooms'] = text
        elif 'ห้อง' in text and not 'ห้องนอน' in text:
            # This is likely bathrooms
            bathroom_match = re.search(r'(\d+)', text)
            if bathroom_match:
                card_data['bathrooms'] = bathroom_match.group(1)
            else:
                card_data['bathrooms'] = text
    
    # Extract property type, listing type, and condition
    tag_box = listing.find('div', class_='box_tag_topic_card')
    if tag_box:
        property_type = tag_box.find('div', class_='box_tag_title')
        if property_type:
            card_data['property_type'] = property_type.text.strip()
        
        listing_type = tag_box.find('span', class_='box_tag_posttype')
        if listing_type:
            card_data['listing_type'] = listing_type.text.strip()
            
        condition = tag_box.find('span', class_='box_tag_condition')
        if condition:
            card_data['condition'] = condition.text.strip()
    
    # Extract detail URL
    detail_link = listing.find('a', class_='image-ratio-4-3')
    if detail_link and detail_link.get('href'):
        card_data['detail_url'] = detail_link['href']
        
        # Extract image URL
        img_elem = detail_link.find('img')
        if img_elem and img_elem.get('src'):
            card_data['image_url'] = img_elem['src']
    
    # Extract last updated time
    last_updated_elem = listing.find('div', class_='istock-lastdate')
    if last_updated_elem:
        card_data['last_updated'] = last_updated_elem.text.strip()
    
    return card_data

def extract_nearby_locations(soup):
    """Extract detailed nearby location information from the nrList element"""
    nearby_locations = []
    
    # Find the nearby locations list
    nearby_list = soup.find('ul', id='nrList')
    if nearby_list:
        # Extract each location item
        location_items = nearby_list.find_all('li', class_='box-link-map')
        
        for item in location_items:
            location_data = {
                'name': None,
                'distance': None,
                'type': None,
                'latitude': None,
                'longitude': None,
                'travel_mode': None,
                'icon': None
            }
            
            # Extract location attributes from data attributes
            location_data['latitude'] = item.get('data-lat')
            location_data['longitude'] = item.get('data-lng')
            location_data['travel_mode'] = item.get('data-mode')
            location_data['icon'] = item.get('data-imgloc')
            
            # Extract location type from data-map attribute
            location_type = item.get('data-map')
            if location_type:
                if 'academy' in location_type:
                    location_data['type'] = 'education'
                elif 'transit' in location_type:
                    location_data['type'] = 'transit'
                elif 'mall' in location_type:
                    location_data['type'] = 'shopping'
                elif 'hospital' in location_type:
                    location_data['type'] = 'healthcare'
                else:
                    location_data['type'] = location_type.replace('living_', '')
            
            # Extract name and distance
            link_elem = item.find('a', class_='box-map-l')
            if link_elem:
                span_elem = link_elem.find('span')
                if span_elem:
                    # Extract text without the img tag
                    img_tag = span_elem.find('img')
                    if img_tag:
                        img_tag.extract()
                    location_data['name'] = span_elem.text.strip()
                
                p_elem = link_elem.find('p')
                if p_elem:
                    location_data['distance'] = p_elem.text.strip()
            
            nearby_locations.append(location_data)
    
    return nearby_locations

def extract_property_details(soup):
    """Extract property details from the detail-property-list elements"""
    property_details = {}
    
    # Find all property detail elements
    detail_elements = soup.find_all('div', class_='detail-property-list')
    
    for element in detail_elements:
        # Get the title/label
        title_elem = element.find('span', class_='detail-property-list-title')
        if not title_elem:
            continue
            
        title = title_elem.text.strip()
        
        # Get the value
        value_elem = element.find('span', class_='detail-property-list-text')
        if not value_elem:
            continue
            
        value = value_elem.text.strip()
        
        # Store in the dictionary with cleaned keys
        key = title.lower().replace(' ', '_').replace('จำนวน', '').replace('ห้อง', 'room')
        property_details[key] = value
    
    return property_details

def extract_listing_detail_data(soup, card_data):
    """Extract detailed information from the property detail page"""
    data = {
        'listing_id': card_data['listing_id'],
        'title': None,
        'price': None,
        'price_per_sqm': None,
        'original_price': None,
        'area': None,
        'floor': None,
        'bedrooms': None,
        'bathrooms': None,
        'room_type': None,  # Added for studio/1-bedroom/etc
        'location': None,
        'project': None,
        'listing_type': card_data['listing_type'] if 'listing_type' in card_data else None,
        'property_type': card_data['property_type'] if 'property_type' in card_data else None,
        'condition': card_data['condition'] if 'condition' in card_data else None,
        'image_url': card_data['image_url'] if 'image_url' in card_data else None,
        'last_updated': None,
        'description': None,
        'images': None,
        'facilities': None,
        'line_contact': None,
        'has_email': False,
        'seller_type': None,
        'posted_date': None,
        'address': None,
        'nearby_locations': None,
        'nearby_locations_detailed': None,
        'nearby_education': None,
        'nearby_transit': None,
        'nearby_shopping': None,
        'nearby_healthcare': None,
        'nearby_other': None,
        'building_features': None,
        'room_features': None,
        'furniture': None,
        'appliances': None,
        'view': None,
        'contact_name': None,
        'contact_phone': None,
        'project_description': None
    }
    
    # Extract title from h1
    title_elem = soup.find('h1', class_='font_sarabun')
    if title_elem:
        data['title'] = title_elem.text.strip()
    
    # Extract price from price-detail element
    price_elem = soup.find('div', class_='price-detail')
    if price_elem:
        price_text = price_elem.find('b')
        if price_text:
            data['price'] = price_text.text.strip()
        
        # Extract price per sqm
        price_per_sqm_elem = price_elem.find('span', class_='price_cal_area_text')
        if price_per_sqm_elem:
            data['price_per_sqm'] = price_per_sqm_elem.text.strip()
    
    # Extract property details from all detail-property-list elements
    detail_elements = soup.find_all('div', class_='detail-property-list')
    for element in detail_elements:
        title_elem = element.find('span', class_='detail-property-list-title')
        value_elem = element.find('span', class_='detail-property-list-text')
        
        if not title_elem or not value_elem:
            continue
            
        title = title_elem.text.strip()
        value = value_elem.text.strip()
        
        if 'พื้นที่ใช้สอย' in title:
            data['area'] = value
        elif 'ชั้น' in title:
            data['floor'] = value
        elif 'ห้องนอน' in title:
            data['bedrooms'] = value
        elif 'ห้องน้ำ' in title:
            data['bathrooms'] = value
        elif 'รูปแบบห้อง' in title:
            data['room_type'] = value
    
    # If we couldn't find bathrooms in the detail elements, try direct extraction
    if not data['bathrooms']:
        bathroom_div = soup.find('div', string=lambda text: text and 'ห้องน้ำ' in text if text else False)
        if bathroom_div:
            bathroom_text = bathroom_div.find_next('span', class_='detail-property-list-text')
            if bathroom_text:
                data['bathrooms'] = bathroom_text.text.strip()
    
    # If we couldn't find bedrooms, check if it's a studio
    if not data['bedrooms']:
        bedroom_div = soup.find('div', string=lambda text: text and 'ห้องสตูดิโอ' in text if text else False)
        if bedroom_div:
            data['bedrooms'] = 'ห้องสตูดิโอ'
            data['room_type'] = 'Studio'
    
    # Extract location and project
    location_elem = soup.find('div', class_='detail-text-zone')
    if location_elem:
        a_tag = location_elem.find('a')
        if a_tag:
            data['location'] = a_tag.text.strip()
    
    project_elem = soup.find('div', class_='detail-text-project')
    if project_elem:
        a_tag = project_elem.find('a')
        if a_tag:
            data['project'] = a_tag.text.strip()
    
    # Extract description
    desc_elem = soup.find('div', class_='wordwrap')
    if desc_elem:
        data['description'] = desc_elem.text.strip()
    
    # Extract images
    gallery = soup.find('div', id='animated-thumbnails')
    if gallery:
        images = []
        for img in gallery.find_all('a', class_='gallery-item'):
            if img.get('data-src'):
                images.append(img['data-src'])
        data['images'] = ', '.join(images) if images else None
    
    # Extract facilities
    facilities = soup.find_all('div', class_='col-xs-6 col-sm-6 col-md-6 d-flex mb-2')
    facility_list = []
    for facility in facilities:
        if facility.find('span', class_='text-additional'):
            facility_list.append(facility.find('span', class_='text-additional').text.strip())
    data['facilities'] = ', '.join(facility_list) if facility_list else None
    
    # Extract contact information
    contact_box = soup.find('div', class_='box_group_chat')
    if contact_box:
        line_elem = contact_box.find('a', class_='co-line')
        if line_elem:
            data['line_contact'] = line_elem.get('data-url')
            
        email_elem = contact_box.find('a', class_='co-email')
        if email_elem:
            data['has_email'] = True
        
        # Contact name and phone
        owner_info = soup.find('div', id='ownerInfo')
        if owner_info:
            name_elem = owner_info.find('div', id='nameOwner')
            if name_elem and name_elem.find('label'):
                data['contact_name'] = name_elem.find('label').text.strip()
    
    # Extract seller type
    seller_type = soup.find('div', class_='detail-text-owner')
    if seller_type:
        data['seller_type'] = seller_type.text.strip()
    
    # Extract posted date and last updated
    date_elems = soup.find_all('div', class_='form-group mb-0 d-md-inline mr_time')
    for elem in date_elems:
        text = elem.text.strip()
        if 'สร้างเมื่อ' in text:
            data['posted_date'] = text.replace('สร้างเมื่อ', '').strip()
        elif 'ดันประกาศล่าสุดเมื่อ' in text:
            data['last_updated'] = text.replace('ดันประกาศล่าสุดเมื่อ', '').strip()
    
    # Extract address
    address_elem = soup.find('div', class_='detail-text-address')
    if address_elem:
        data['address'] = address_elem.text.strip()
    
    # Extract detailed nearby locations
    nearby_locations = extract_nearby_locations(soup)
    
    # Store the full detailed data
    if nearby_locations:
        # Convert to JSON string for Excel storage
        data['nearby_locations_detailed'] = json.dumps(nearby_locations, ensure_ascii=False)
        
        # Create categorized nearby location lists
        education_locations = []
        transit_locations = []
        shopping_locations = []
        healthcare_locations = []
        other_locations = []
        
        # Simple formatted list for Excel
        all_nearby = []
        
        for location in nearby_locations:
            if location['name'] and location['distance']:
                formatted_location = f"{location['name']} ({location['distance']})"
                all_nearby.append(formatted_location)
                
                if location['type'] == 'education':
                    education_locations.append(formatted_location)
                elif location['type'] == 'transit':
                    transit_locations.append(formatted_location)
                elif location['type'] == 'shopping':
                    shopping_locations.append(formatted_location)
                elif location['type'] == 'healthcare':
                    healthcare_locations.append(formatted_location)
                else:
                    other_locations.append(formatted_location)
        
        data['nearby_locations'] = ', '.join(all_nearby) if all_nearby else None
        data['nearby_education'] = ', '.join(education_locations) if education_locations else None
        data['nearby_transit'] = ', '.join(transit_locations) if transit_locations else None
        data['nearby_shopping'] = ', '.join(shopping_locations) if shopping_locations else None
        data['nearby_healthcare'] = ', '.join(healthcare_locations) if healthcare_locations else None
        data['nearby_other'] = ', '.join(other_locations) if other_locations else None
    
    # Extract building and room features
    features_section = soup.find('div', class_='box-features')
    if features_section:
        building_features = []
        room_features = []
        furniture = []
        appliances = []
        view = []
        
        # Try to categorize features
        feature_items = features_section.find_all('div', class_='feature-item')
        for item in feature_items:
            feature_text = item.text.strip()
            if any(keyword in feature_text.lower() for keyword in ['อาคาร', 'ลิฟท์', 'ล็อบบี้', 'สระว่ายน้ำ', 'ฟิตเนส']):
                building_features.append(feature_text)
            elif any(keyword in feature_text.lower() for keyword in ['ห้อง', 'ระเบียง', 'กระจก']):
                room_features.append(feature_text)
            elif any(keyword in feature_text.lower() for keyword in ['โซฟา', 'เตียง', 'ตู้', 'โต๊ะ', 'เก้าอี้']):
                furniture.append(feature_text)
            elif any(keyword in feature_text.lower() for keyword in ['ทีวี', 'ตู้เย็น', 'เครื่องซักผ้า', 'เครื่องปรับอากาศ']):
                appliances.append(feature_text)
            elif any(keyword in feature_text.lower() for keyword in ['วิว', 'ทิวทัศน์', 'แม่น้ำ', 'สวน']):
                view.append(feature_text)
        
        data['building_features'] = ', '.join(building_features) if building_features else None
        data['room_features'] = ', '.join(room_features) if room_features else None
        data['furniture'] = ', '.join(furniture) if furniture else None
        data['appliances'] = ', '.join(appliances) if appliances else None
        data['view'] = ', '.join(view) if view else None
    
    # Extract project description
    project_desc_elem = soup.find('div', class_='box-show-text-all-project')
    if project_desc_elem:
        data['project_description'] = project_desc_elem.text.strip()
    
    return data

def get_total_pages(driver):
    try:
        pagination = driver.find_element(By.CLASS_NAME, "pagination")
        pages = pagination.find_elements(By.TAG_NAME, "li")
        if pages:
            last_page = pages[-2].text  # Last page number is usually second to last element
            return int(last_page)
    except Exception as e:
        print(f"Error getting total pages: {str(e)}")
        return 1
    return 1

def save_progress(data, filename='property_listings_progress.xlsx'):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"Progress saved to {filename}")

def main():
    # For scraping multiple listings
    base_url = "https://www.livinginsider.com/searchword/Condo/all/1/รวมประกาศขาย-เช่าคอนโด.html"
    #base_url = "https://www.livinginsider.com/searchword/Home/all/1/รวมประกาศขาย-เช่าบ้าน.html"
    #base_url = "https://www.livinginsider.com/searchword/Townhome/all/1/รวมประกาศขาย-เช่าทาวน์เฮ้าส์-ทาวน์โฮม.html"
    #base_url = "https://www.livinginsider.com/searchword/Land/all/1/รวมประกาศขาย-เช่าที่ดิน.html"
    #base_url = "https://www.livinginsider.com/searchword/Commercial/all/1/รวมประกาศขาย-เช่า-เซ้งตึกแถว-อาคารพาณิชย์.html"
    #base_url = "https://www.livinginsider.com/searchword/Hotel_apartment/all/1/รวมประกาศขาย-เช่า-เซ้งกิจการ-โรงแรม-หอพัก-อพาร์ตเมนต์.html"
    #base_url = "https://www.livinginsider.com/searchword/Showroom/all/1/รวมประกาศขาย-เช่า-เซ้งโชว์รูม-สํานักงานขาย.html"
    #base_url = "https://www.livinginsider.com/searchword/Salesarea/all/1/รวมประกาศขาย-เช่า-เซ้งพื้นที่ขายของ.html"
    #base_url = "https://www.livinginsider.com/searchword/Homeoffice/all/1/รวมประกาศขาย-เช่า-เซ้งโฮมออฟฟิศ.html"
    #base_url = "https://www.livinginsider.com/searchword/Officespace/all/1/รวมประกาศขาย-เช่า-เซ้งสำนักงาน.html"
    
    driver = setup_driver()
    all_data = []
    
    try:
        print("Loading first page...")
        driver.get(base_url)
        wait = WebDriverWait(driver, 10)
        
        # Wait for search results to load
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "istock-list")))
        
        # Get total number of pages
        total_pages = get_total_pages(driver)
        print(f"Total pages to scrape: {total_pages}")
        
        # Limit to first 3 pages for testing
        max_pages = min(3, total_pages)
        
        # Loop through pages
        for page in range(1, max_pages + 1):
            print(f"Processing page {page}/{max_pages}")
            
            # Load page
            if page > 1:
                page_url = base_url.replace("/1/", f"/{page}/")
                driver.get(page_url)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "istock-list")))
            
            # Get all listings on current page
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Find all listing containers
            listing_containers = soup.find_all('div', class_='padding_topic')
            
            print(f"Found {len(listing_containers)} listing containers on page {page}")
            
            # Limit to first 5 listings per page for testing
            max_listings = min(60, len(listing_containers))
            
            for container_idx, container in enumerate(listing_containers[:max_listings], 1):
                # Find the actual listing within the container
                listing = container.find('div', class_='istock-list')
                if not listing:
                    continue
                
                try:
                    # Extract basic data from listing card
                    card_data = extract_listing_card_data(listing)
                    listing_id = card_data['listing_id']
                    
                    print(f"Processing listing {container_idx} on page {page} (ID: {listing_id})")
                    
                    if card_data['detail_url']:
                        # Visit detail page
                        detail_url = card_data['detail_url']
                        print(f"Visiting: {detail_url}")
                        driver.get(detail_url)
                        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "detail-content")))
                        
                        # Extract detailed information
                        detail_soup = BeautifulSoup(driver.page_source, 'html.parser')
                        listing_data = extract_listing_detail_data(detail_soup, card_data)
                        all_data.append(listing_data)
                        
                        # Save progress after every 5 listings
                        if len(all_data) % 5 == 0:
                            save_progress(all_data, 'property_listings_progress.xlsx')
                        
                        time.sleep(1)  # Be nice to the server
                except Exception as e:
                    print(f"Error processing listing {listing_id if 'listing_id' in locals() else 'unknown'}: {str(e)}")
                    continue
            
            # Save progress after each page
            save_progress(all_data, 'property_listings_progress.xlsx')
        
        # Final save to main file
        save_progress(all_data, 'property_listings_final.xlsx')
            
        print(f"Completed! Total listings scraped: {len(all_data)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Save whatever data we have
        if all_data:
            save_progress(all_data, 'property_listings_error.xlsx')
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
