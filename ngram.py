from nltk import ngrams

def split_into_n_gram(sentence, n=1):
    split_text = ngrams(sentence.split(), n)
    ngram_result = []
    for grams in split_text:
        text = ''
        for value in grams:
            text = ' '.join([text, str(value)])
        ngram_result.append(text)
    return ngram_result

def split_into_multiple_gram(sentence):
    unigram = split_into_n_gram(sentence, 1)
    bigram = split_into_n_gram(sentence, 1)
    trigram = split_into_n_gram(sentence, 1)

    return unigram + bigram + trigram

