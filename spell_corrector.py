'''
Spell checking the search_queries. It takes a rather long time to run;
suitable for over-the-night runs.

INPUT FILES: 
train.csv (raw data file)
test.csv  (raw data file)

OUTPUTS:
spell_corr.py

__Author__:
Ali Narimani

__Veresion__:
1.2
'''

import requests
import re
import time
from random import randint
import pandas as pd

# Reading input data:
train = pd.read_csv('../data/train.csv', encoding="ISO-8859-1")
test = pd.read_csv('../data/test.csv', encoding="ISO-8859-1")

START_SPELL_CHECK="<span class=\"spell\">Showing results for</span>"
END_SPELL_CHECK="<br><span class=\"spell_orig\">Search instead for"

HTML_Codes = (
		("'", '&#39;'),
		('"', '&quot;'),
		('>', '&gt;'),
		('<', '&lt;'),
		('&', '&amp;'),
)

def spell_check(s):
	q = '+'.join(s.split())
	time.sleep(  randint(0,2) ) #relax and don't make google angry
	r = requests.get("https://www.google.co.uk/search?q="+q)
	content = r.text
	start=content.find(START_SPELL_CHECK) 
	if ( start > -1 ):
		start = start + len(START_SPELL_CHECK)
		end=content.find(END_SPELL_CHECK)
		search= content[start:end]
		search = re.sub(r'<[^>]+>', '', search)
		for code in HTML_Codes:
			search = search.replace(code[1], code[0])
		search = search[1:]
	else:
		search = s
	return search ;


### start the spell_check :
data = ['train','test']
outfile = open("spell_corr.py", "w")
outfile.write('spell_check_dict={\n')


for df in data:
    searches = eval(df)[10:20].search_term
 
    for search in searches:
        speel_check_search= spell_check(search)
        if (speel_check_search != search):
            outfile.write('"'+search+'"'+" : " + '"'+ speel_check_search+'"'+', \n')
            
outfile.write('}')
outfile.close()

# End of code
