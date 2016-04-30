.IGNORE:

all: testutils

install:
	pip install -r requirements.txt
	python -m nltk.downloader all


testutils:
	@python nlp_utils.py > /dev/null && echo "nlp_utils works!"
	@echo "================================="
	@python ngrams.py > /dev/null && echo "ngrams works!"
	@echo "================================="

clean:
	/bin/rm *.pyc 2>&1 > /dev/null
