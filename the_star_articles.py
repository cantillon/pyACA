#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:18:16 2020
@author: cantillon
Web scraper for the-star.co.ke
"""
def the_star_scrape(input_file, search_term):
    from lxml import html
    import requests
    from nltk import FreqDist
    import pandas as pd
    articles = []
    num = 0
    pages_found = 0
    terms_found = 0
    with open(input_file) as links:
        for link in links:
            num += 1
            link = link.strip()
            htmlsource = requests.get(link).text
            if '<h1>404</h1><h4>Page not found.</h4>' in htmlsource:
                page_found = False
                articles.append({ 'num' : num, 'page found' : page_found})
            else:
                page_found = True
                pages_found += 1
                tree = html.fromstring(htmlsource)
                secondary_title = tree.xpath('//h2[@class = "article-title article-title-secondary"]//text()')
                secondary_title = ' '.join(item2 for item3 in [item1.split() for item1 in secondary_title] for item2 in item3)
                primary_title = tree.xpath('//h1[@class = "article-title article-title-primary"]//text()')
                primary_title = ' '.join(item2 for item3 in [item1.split() for item1 in primary_title] for item2 in item3)
                summary = tree.xpath('//div[@class = "article-intro"]//text()')
                summary = ' '.join(item2 for item3 in [item1.split() for item1 in summary] for item2 in item3)
                author = tree.xpath('//span[@class = "authors-list"]//text()')
                try:
                    author = author[0].strip()
                except:
                    author = author
                date = tree.xpath('//div[@class = "article-published"]//text()')
                date = date[0].strip()
                text = tree.xpath('//div[@class = "text"]//text()')
                text = ' '.join(item2 for item3 in [item1.split() for item1 in text] for item2 in item3)
                if search_term in text:
                    term_found = True
                    terms_found += 1
                else:
                    term_found = False
                fd = FreqDist(text.split(' '))[search_term]
                articles.append({ 'num' : num, 'page found' : page_found, 'term found' : term_found, 'freq dist' : fd, 'author' : author, 'date' : date, 'secondary title' : secondary_title, 'primary title' : primary_title, 'summary' : summary, 'text' : text, 'link' : link})
    df = pd.DataFrame(articles)
    df.to_csv('%s_output.csv' % input_file[:-4], encoding='utf-8', index=False)
    print(pages_found, 'pages found, out of', num, 'links searched. In', terms_found, 'pages, the search term "%s"' % search_term, 'was found')

the_star_scrape('/home/cantillon/Dropbox/ACA/code/the_star_news.csv','abortion')

the_star_scrape('/home/cantillon/Dropbox/ACA/code/the_star_opinion.csv','abortion')

# testing git
# testing again
# TODO create new list of authors
# TODO create new list of publication years

# TODO when the script ends, print the number of articles scraped and some additional info: how many authors, from which years, etc.
# TODO after doing the sentiment analysis and the lda, append the results to the dictionary and the dataframe/csv file, in order to link those results with other variables, perform further analyses and visualize the results of those analyses.