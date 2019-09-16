

import requests
from bs4 import BeautifulSoup
from queue import Queue
from IPython import embed
from pathlib import Path
import re


import time

BASE_URL = 'http://tobacco.stanford.edu/tobacco_web/images/tobacco_ads/'

#BASE_URL = 'http://tobacco.stanford.edu/tobacco_web/images/tobacco_ads/global_village/asian/'



def scrape_ads():

    urls_to_process = Queue()
    urls_to_process.put(BASE_URL)
    urls_processed = set()

    while True:

        if urls_to_process.qsize() == 0:
            break

        print("queue len", urls_to_process.qsize())
        time.sleep(3)
        url = urls_to_process.get()
        print(url)

        req = requests.get(url)


        if url[-1] == '/':
            soup = BeautifulSoup(req.text, 'html.parser')
            for link in soup.find_all('a'):
                link_url = f'http://tobacco.stanford.edu{link.get("href")}'
                if (
                        link.contents[0] == '[To Parent Directory]' or
                        re.match(r'.+\/small\/[a-z_0-9]+\.jpg', link_url) or
                        re.match(r'.+\/medium\/[a-z_0-9]+\.jpg', link_url)
                ):
                    continue

                else:

                    urls_to_process.put(link_url)
        else:

            # only download large but not small/medium images
            if re.match(r'.+\/large\/[a-z_0-9]+\.jpg', url):

                img_class = url.split('/')[-4]
                img_subclass = url.split('/')[-3]
                filename = url.split('/')[-1]

                p = Path(img_class, img_subclass, filename)
                p.parent.mkdir(exist_ok=True, parents=True)
                with open(p, 'wb') as f:
                    f.write(req.content)

            else:
                print("not downloading: ", url)






if __name__ == '__main__':
    scrape_ads()