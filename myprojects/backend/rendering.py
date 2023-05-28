import requests
from flask import Flask, render_template
from bs4 import BeautifulSoup
from urllib.parse import urljoin

app = Flask(__name__)

@app.route('/')
def hello():
    html_url = "https://roaring-manatee-eb9a2f.netlify.app/login.html#"  # 외부 HTML 주소
    response = requests.get(html_url)  # HTML 가져오기
    html_content = response.text  # HTML 내용 추출
    
    # BeautifulSoup을 사용하여 HTML 파싱
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # HTML 내에서 상대 경로를 절대 경로로 변환
    base_url = response.url
    for tag in soup.find_all('a', href=True):
        tag['href'] = urljoin(base_url, tag['href'])
    for tag in soup.find_all('img', src=True):
        tag['src'] = urljoin(base_url, tag['src'])
    for tag in soup.find_all('link', href=True):
        tag['href'] = urljoin(base_url, tag['href'])
    for tag in soup.find_all('script', src=True):
        tag['src'] = urljoin(base_url, tag['src'])
    
    # 변환된 HTML 내용을 반환
    return str(soup)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)