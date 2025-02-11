## Download GloVe embeedings: https://nlp.stanford.edu/projects/glove/

wget https://nlp.stanford.edu/data/glove.6B.zip

unzip glove.6B.zip -d ./assets/

conda env create -f environment.yml

conda activate sentiment_env
