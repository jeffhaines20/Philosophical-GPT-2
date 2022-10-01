import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

st.title('Artificial Philosopher')
st.sidebar.image('school_of_athens.jpg', use_column_width=True)

st.write('Uses GPT-2 and the Stanford Encyclopedia of Philosophy to generate answers to philosophical questions.')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained('gpt2')

text = st.text_input('Enter a philosophical question.')

if text:
    query = ('+').join(text.split())

    # query the SEP
    url = 'https://plato.stanford.edu/search/searcher.py?query='+query

    html = urlopen(url)
    bs = BeautifulSoup(html, 'html.parser')

    #grab the first result
    page = bs.find_all('a',{'class':'l'})[0].attrs['href']
    page = page.split('&page')[0]

    # open the result
    html = urlopen(page)
    soup = BeautifulSoup(html, 'html.parser')

    # get answer
    p_tags = soup.find_all('p')
    text = [tag.text for tag in p_tags]
    text = ''.join(text)
    text = text.split('https')[0].replace('\n',' ').strip()

    # take the first five sentences to feed into GPT-2
    text = text.split('. ')
    text = text[:15]
    text = ('. ').join(text) + '.' 

    # send the SEP text to GPT-2
    text_ids = tokenizer.encode(text, return_tensors = 'pt')

    encoding = model.generate(text_ids, max_length= 1000, do_sample = True, temperature = 1.)

    # showing only the generated text
    answer = tokenizer.decode(encoding[0], skip_special_tokens=True)[len(text):]

    # there is a problem with the model giving sentence fragments. Reove the first and last sentence.
    answer = answer.split('.')
    answer = answer[1:-1]
    answer = ('.').join(answer) + '.'

    st.write(answer)

else:
    st.write('You have not asked a question yet.')