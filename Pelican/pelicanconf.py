#!/usr/bin/env python
# -*- coding: utf-8 -*- #

AUTHOR = 'Cody Fernandez'
SITENAME = 'Hoopyeah'
SITEURL = ''
THEME = '../../pelican-themes/gum'
OUTPUT_PATH = '../'

TYPOGRIFY = True

PATH = 'content'
STATIC_PATHS = ['extra/CNAME', 'extra/robots.txt']
EXTRA_PATH_METADATA = {'extra/CNAME': {'path': 'CNAME'},
                       'extra/robots.txt': {'path': 'robots.txt'},
                       }

TIMEZONE = 'America/New_York'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),
         ('Python.org', 'https://www.python.org/'),
         ('Jinja2', 'https://palletsprojects.com/p/jinja/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

GITHUB_URL = ''
TWITTER_URL = ''
FACEBOOK_URL = ''
GOOGLEPLUS_URL = ''
SITESUBTITLE = 'Jesus Christ is King of the Universe'

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True