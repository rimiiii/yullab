import time
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd
import cv2
import numpy as np

def get_image_from_soup(soup):
    image_url = soup.find('meta', {'property': 'og:image'})['content']
    response = requests.get(image_url, stream=True).raw
    img = np.asarray(bytearray(response.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def get_author(soup):
    try:
        return soup.find('div', class_='author').find('a').text
    except:
        return None
    
def get_publisher(soup):
    try:
        return soup.find('div', class_='prod_info_text').find('a').text
    except:
        return None
    
def get_publish_date(soup):
    try:
        return soup.find('div', class_='publish_date').contents[-1].replace('·', '').strip()
    except:
        return None
    
def get_score(soup):
    try:
        return float(soup.find('span',{'class': 'review_score feel_lucky'}).text)
    except:
        return None
    
def get_categories(soup):
    try:
        categories_soup = soup.find_all('a', class_='intro_category_link')
        return [category.text for category in categories_soup]
    except:
        return None
    
def get_info_text(soup):
    try:
        info_soup = soup.find_all('div', class_='info_text')
        info_text_header = info_soup[0].text
        info_text = info_soup[1].text
        return info_text_header, info_text
    except:
        return None
    
def get_author_info(soup):
    try:
        return soup.find('p', {'class': 'info_text'}).text
    except:
        return None
    
def get_image(soup):
    try:
        return get_image_from_soup(soup)
    except:
        return None
    
def get_author_prod_names(soup):
    try:
        author_prod_names = soup.find_all('span',{'class': 'prod_name'})
        return [prod_name.text for prod_name in author_prod_names]
    except:
        return None


if __name__ == '__main__':
    # base_prod_id = 201507091
    base_prod_id = 201505890
    prod_id = base_prod_id
    for idx in tqdm(range(base_prod_id)):
        url_prod_id = f'S000{str(prod_id).zfill(len(str(base_prod_id)))}'
        req = requests.get(f'https://product.kyobobook.co.kr/detail/{url_prod_id}')
        html = req.text
        soup = BeautifulSoup(html, 'html.parser')
        book_info = [prod_id, get_author(soup), get_author_info(soup), get_author_prod_names(soup), get_categories(soup), get_image(soup), get_info_text(soup), get_publish_date(soup), get_publisher(soup), get_score(soup)]
        prod_id -= 1
        mode = 'w' if idx == 0 else 'a'
        pd.DataFrame(book_info).T.to_csv('book_info.csv', header=False, mode='w' if idx == 0 else 'a')