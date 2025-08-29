import scrapy
from bs4 import BeautifulSoup


class BlogSpider(scrapy.Spider):
    name = 'narutospider'
    start_urls = ['https://naruto.fandom.com/wiki/Category:Jutsu']


    def parse_jutsu(self,response):
        jutsu_name=response.css('span.mw-page-title-main::text').extract()[0]
        jutsu_name=jutsu_name.strip()

        div_selector= response.css('div.mw-parser-output')[0]
        div_html=div_selector.extract()

        soup=BeautifulSoup(div_html).find('div')


        jutsu_type=''
        if soup.find('aside'):
            aside=soup.find('aside')

            for cell in aside.find_all('div',{'class' :'pi-data'}):
                if cell.find('h3'):
                    cell_name=cell.find('h3').text.strip()
                    if cell_name == 'Classification':
                        jutsu_type=cell.find('div').text.strip()



        soup.find('aside').decompose()

        jutsu_description = soup.text.strip()
        jutsu_description=jutsu_description.split('Trivia')[0].strip()


        return dict(
            jutsu_name=jutsu_name,
            jutsu_type=jutsu_type,
            jutsu_description=jutsu_description
        )


    def parse(self, response):
        # extract all jutsu links
        for href in response.css("a.category-page__member-link::attr(href)").getall():
            yield response.follow(href, self.parse_jutsu)

        # pagination (next page button)
        next_page = response.css("a.category-page__pagination-next::attr(href)").get()
        if next_page:
            yield response.follow(next_page, self.parse)



