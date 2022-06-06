# !pip install transformers[sentencepiece]
# pip install -U spacy
# !python spacy download fr_core_news_sm


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

from transformers import pipeline

import spacy
from spacy import displacy
from spacy.matcher import Matcher

from collections import Counter
import random # to use color_func
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd

import torch

# modele d'analyse de sentiments bert
multilang_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# analyse de mots avec le modele d'articles de presse français (small)
nlp = spacy.load("fr_core_news_sm")

# ajouts de stopwords a la liste existante
stop_words = stopwords.words("french") + ['ce','cette','après','très','ça','avoir', 'près','a','comme', 'cet', 'tout', 'toute']

# création de la liste de mots via un pattern spacy 
matcher = Matcher(nlp.vocab)

# Écris un motif pour un nom suivi d'un ou deux adjectif
pattern = [{"POS": "NOUN"}, {"POS": "ADJ"}, {"POS": "ADJ", "OP": "?"}]

# Ajoute le motif au matcher et applique le matcher au doc
matcher.add("ADJ_NOUN_PATTERN", [pattern])



# analyse de sentiment par phrase au sein de chaque article (permet analyse en fonction du contexte)

def sentiment_analysis(df_test, column='Content'):
    """
    Define positive, negative and neutral sentences in a text of less than 512 caracters.
    Tokenizing in sentences with nltk, predicting sentiment with Bert and Hugging Face.

    input : df_test(dataframe) = dataframe in which is store the text to analyse
            column(string) = name of the column in which is store the text to analyse
    output : dataframe (with 3 new columns added, each containing positive sentences, negative sentences, neutral sentences)
    """

    # on decide de conserver les phrases analysees en fonction de leur sentiment  

    df_test['stars_neg'] = ""
    df_test['stars_pos'] = ""
    df_test['stars_neut'] = ""

    for row in range(len(df_test)) : 
        text = df_test[column][row]

        sent_text = nltk.sent_tokenize(text)
        list_neg = []
        list_pos = []
        list_neut = []

        for phrase in sent_text:
            # si la phrase est superieure à 512 caractères elle ne pourra pas être analysé par notre modele
            # qui est limité en nombre de caractere
            if len(phrase) <= 512:
                score = multilang_classifier([phrase])
                # le modele associe un accuracy score à chaque prediction, on ne conserve pas de prediction
                # en dessous de 0.3
                if score[0]['score'] >= 0.3:
                    # on détermine comme positive une phrase ayant été prédite avec 4 ou 5 étoiles
                    if (score[0]['label'] in ['5 stars','4 stars']):
                        list_pos.append(phrase)
                    # on détermine comme positive une phrase ayant été prédite avec 1 ou 2 étoiles
                    elif score[0]['label'] in ['1 star','2 stars']:
                        list_neg.append(phrase)
                    # sinon la phrase est neutre (3 étoiles)
                    else : 
                        list_neut.append(phrase)

        # on alimente le dataframe avec nos resultats
        df_test['stars_neg'][row] = list_neg

        df_test['stars_pos'][row] = list_pos

        df_test['stars_neut'][row] = list_neut

    return df_test


# analyse de sentiment general de l'article

def final_sentiment(df_test):
    """
    Define global sentiment of a text using prior results from function sentiment_analyses. 
    Which means that the text has already been divided in sentences dispatched by positive, negative, and neutral feeling.
    
    input : df_test(dataframe) = dataframe in which positive, negative and neutral sentences are stored respectively in columns named 'stars_pos', 'stars_neg', 'stars_neut'.
    output : dataframe (with a new column added, contaning final sentiment ('positif' for positive, 'negatif' for negative, 'neutre' for neutral, or 'non renseigné' is sentiment not found))
    """

    df_test['final_sentiment_hugging'] = 'non renseigné'

    # pour chaque article on détermine son sentiment général en fonction de la proportion de 
    # phrases négatives, positives et neutres dans l'article entier

    for i in range(len(df_test)):
        # on passe les listes de phrases positives, negatives et neutres sous forme de texte
        # afin de vérifier leur longueur
        positive_length_text = len(' '.join(df_test['stars_pos'][i]))
        negative_length_text = len(' '.join(df_test['stars_neg'][i]))
        neutral_length_text = len(' '.join(df_test['stars_neg'][i]))

        # on verifie la longueur de l'article entier
        content = df_test['Content'][i]
        full_length_text = len(content)

        # si l'article est vide ou mentionne que le contenu n'a pu être récupéré
        # on ne l'analyse pas
        if full_length_text != 0 and 'contenu non récupéré' not in content :
            
            # si un article contient au moins 80% de phrases positives, il est positif
            if (positive_length_text / full_length_text) >= 0.8:  
                df_test['final_sentiment_hugging'][i] = 'positif'

            # si un article contient au moins 20% de phrases negatives, il est negatif
            # les parties negatives peuvent occuper peu de place dans un article de presse 
            # car le ton est en general modéré, ou les parties positives et negatives s'équilibrent, 
            # mais les phrases negatives qui ont pu etre determinees comme tel par notre modele
            # sont des phrases qui ont un impact fort sur le sentiment general de l'article
            elif (negative_length_text / full_length_text) >= 0.2:
                df_test['final_sentiment_hugging'][i] = 'negatif'
            
            # si ces conditions ne sont pas remplies, on peut affiner l'analyse, et ne plus 
            # prendre en compte la longueur de l'article entier
            else:
                # on prend en compte la part de neutre de l'article pour voir si cela
                # fait pencher la balance d'un cote ou de l'autre du positif ou negatif
                if (negative_length_text + neutral_length_text) > positive_length_text :
                    df_test['final_sentiment_hugging'][i] = 'negatif'
                elif (positive_length_text + neutral_length_text) > negative_length_text :
                    df_test['final_sentiment_hugging'][i] = 'positif'
                # si besoin on affine d'avantage en regardant s'il y a plus de positif ou de negatif 
                elif positive_length_text > negative_length_text :
                    df_test['final_sentiment_hugging'][i] = 'positif'
                elif positive_length_text < negative_length_text :
                    df_test['final_sentiment_hugging'][i] = 'negatif'
                # enfin si aucune de ces conditions n'est remplie c'est que l'article est neutre
                # le but étant d'avoir le moins possible d'articles neutre, les parties negatives peuvent occuper peu de place
                # dans un article de presse car le ton est en general modéré mais elles ont un impact fort sur le sentiment
                # general de l'article
                else :
                    df_test['final_sentiment_hugging'][i] = 'neutre'

    return df_test

# fonction de coloration de mots en couleur vert et rouge pour les nuages de mots des parties positives et negatives de l'article

def green_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(146,53%%, %d%%)" % random.randint(40,54)

def red_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0,94%%, %d%%)" % random.randint(38, 63)


# fonction nettoyage de l'article en enlevant les stopwords et ne conservant que les mots

def clean(texte):
    """ 
    Tokenize a text and remove stopwords and numeric data.

    input : texte(str) = text to clean.
    output : string (cleaned text)
    """
    tokens = nlp(texte.lower())
    tokens_clean = [token.text for token in tokens if (token.text not in stop_words) and (token.is_alpha)]

    # une autre option est de ne conserver que les lemma, 
    #tokens_lemm = [token.lemma_ for token in tokens_clean]

    return tokens_clean

# creation d'une liste de mots/expressions a utiliser pour le wordcloud

def wordcloud_resume_list(article, nb_most_commons = 30):
    """
    Looking for pattern, entities, and most commons words, in a given text, to create a list of words that can be used to make a wordcloud.
    Using Spacy and nltk.

    input : article(str) = text to analyse.
            nm_most_communs(int) = number of most commons words to keep in our list.
    output : list of words/expressions.
    """
    # création de la liste de mots/expressions  via un pattern spacy :

    # création d'un outil pour retrouver les match d'un motif dans un texte
    # matcher = Matcher(nlp.vocab)

    # creation du motif d'un nom suivi de un ou deux adjectifs
    # pattern = [{"POS": "NOUN"}, {"POS": "ADJ"}, {"POS": "ADJ", "OP": "?"}]

    # ajout du motif à notre outil matcher
    # matcher.add("ADJ_NOUN_PATTERN", [pattern])

    # le modele retourne tous les match obtenu en fonction du pattern créé
    doc = nlp(article)
    matches = matcher(doc)

    # parmi ces match on ajoute la portion de texte concernée à notre liste de mots/expressions
    list_matches_pattern = []
    for match_id, start, end in matches:
        list_matches_pattern.append(doc[start:end].text)


    # création de la liste de mots via les entities spacy :

    # passer les début de phrase en lower pour ne pas fausser la reconnaissance d'entities
    # on en profite pour nettoyer le texte des residus du scraping (parenthèses ou caractères mal positionnés)
    text_low = re.sub(r'(\.\s+)([A-Z])', lambda match : ' '+match.group(2).lower(), article)
    text_low2 = re.sub(r'(\)\s*)([A-Z])', lambda match : ') '+match.group(2).lower(), text_low)
    text_low3 = re.sub(r'(\w+\’)|(\w+\')|(\s+)', ' ', text_low2)
    text1= nlp(text_low3)

    # on cherche les entites reconnues par Spacy dans ce texte, et on les ajoute à notre liste
    list_words_ents = []
    for word in text1.ents:
        list_words_ents.append(word.text)


    # création de la liste de mots via frequencies most commons

    article_clean = clean(article)  # appeler de la fonction clean pour nettoyer le texte
    list_freq = nltk.FreqDist(article_clean).most_common(nb_most_commons)

    list_words_freq = []
    for i in list_freq:
        # si un mot contient une majuscule il aura ete reconnu comme entite par Spacy et est deja contenu de maniere
        # plus pertinente (par exemple prénom suivi du nom) dans la liste de mots list_words_ents
        if i[0][0].islower():
            list_words_freq.append(i[0])

    # on regroupe toutes les les listes en une seule, qui servira de base pour créer un nuage de mots par la suite
    list_cloud_text = list_matches_pattern + list_words_ents + list_words_freq
    if len(list_cloud_text) == 0:
        list_cloud_text.append("...")
        
    return list_cloud_text

# creation d'un visuel avec deux wordcloud : un pour les éléments positif (vert), un pour les éléments négatifs (rouge)

def wordclouds_stars(df, row, keyword, keyword2):
    """
    Looking for pattern, entities, and most commons words, in a given text, to create a list of words that can be used to make a wordcloud (with clean and wordcloud_resume_list functions).
    This process is made for positive sentences and for negative sentences, so as to plot two different wordclouds (green for positive words/expressions, red for negative word/expressions).
    Using Spacy, nltk and WordCloud.

    ! Warning ! : you first need to run green_color_func, red_color_func, clean and wordcloud_resume_list (in sentiments_resume) to use this function.

    input : df(dataframe) = dataframe in which is the text to analyse.
            row(int) = number of row of the dataframe in which is the text to analyse.
            keyword(string) = keyword used for the search of the press articles, to remove from wordclouds.
            keyword2(string) = keyword used for the search of the press articles, to remove from wordclouds.
    output : plot with two wordclouds (green : positive words/expressions, red : negative words/expressions)
    """
    # on ne souhaite pas que les mots clés de la recherche soient affichés, ils ne doivent pas etre interpretables 
    # comme elements positifs ou negatifs, ils sont presents dans les deux parties car mentionnés dans les phrases
    # qu'elles soient negatives ou positives
    stop_words_keywords = [keyword, keyword2]

    # partie positive : 
 
    text_pos = ' '.join(df['stars_pos'][row])
    # condition de securite si l'article est vide cela ne bloquera pas le processus
    if text_pos == '':
        list_cloud_text_pos=['...']
    else:
        list_cloud_text_pos = wordcloud_resume_list(text_pos)

    word_cloud_dict_pos = Counter(list_cloud_text_pos)
    
    wordcloud_pos = WordCloud(background_color='white', collocations=False, width=480, stopwords = stop_words_keywords,
                              height=480, max_font_size=200, min_font_size=10).generate_from_frequencies(word_cloud_dict_pos)

    # partie negative : 

    text_neg = ' '.join(df['stars_neg'][row])
    # condition de securite si l'article est vide cela ne bloquera pas le processus
    if text_neg == '':
        list_cloud_text_neg=['...']
    else:
        list_cloud_text_neg = wordcloud_resume_list(text_neg)

    word_cloud_dict_neg = Counter(list_cloud_text_neg)
    wordcloud_neg = WordCloud(background_color='white', collocations=False, width=480,stopwords = stop_words_keywords,
                              height=480, max_font_size=200, min_font_size=10).generate_from_frequencies(word_cloud_dict_neg)

    # plot avec deux wordcloud : 

    fig, ax = plt.subplots(figsize = (20,10))

    # Définition du premier graphique
    ax1 = plt.subplot(121) 
    ax1.imshow(wordcloud_pos.recolor(color_func=green_color_func),
            interpolation="bilinear")
    ax1.axis("off")
    ax1.margins(x=0, y=0)

    # Définition du second graphique
    ax2 = plt.subplot(122) 
    ax2.imshow(wordcloud_neg.recolor(color_func=red_color_func),
            interpolation="bilinear")
    ax2.axis("off")
    ax2.margins(x=0, y=0)

    return plt.show()


# création du résumé de l'article, permettant de n'afficher que les parties concernant le premier clé

def resume(text, keyword):
    """
    Make a resume of the text arount the chosen them/entity.

    Tokenizing text in sentences, loop on all sentences of the text and : 
    return the sentence just before a sentence containing the keyword, the sentence containing the keyword, and the sentence just after a sentence containing the keyword.
    Remove duplicates and keep chronogical order of the reading of the text.

    input : text(str) = text for which you need a resume
            keyword(str) = keyword to look for in sentences.
    output : string (resume of the text)
    """

    # on découpe l'article en phrases
    sent_text = nltk.sent_tokenize(text)

    resume_article = []

    # pour chaque phrase, si elle contient le premier mot clé saisi par l'utilisateur (mot clé fort)
    # elle est ajoutée au résumé, ainsi que la phrase la précédent et la phrase la suivant 
    # (permet de conserver la contextualisation)
    for i in range(0,len(sent_text)):  
        if keyword in sent_text[i] :
            if i >= 1 :
                if len(sent_text) >= i+2:
                    resume_article.extend([sent_text[i-1], sent_text[i], sent_text[i+1]])

                elif len(sent_text) == i+1: 
                    resume_article.extend([sent_text[i-1], sent_text[i]])

            elif i < 1 : 
                if len(sent_text) >= i+2:
                    resume_article.extend([sent_text[i], sent_text[i+1]])

                elif len(sent_text) == i+1: 
                    resume_article.append(sent_text[i])

            elif i < 2 : 
                if len(sent_text) >= i+2:
                    resume_article.extend([sent_text[i-1], sent_text[i], sent_text[i+1]])

                elif len(sent_text) == i+1:
                    resume_article.extend([sent_text[i-1], sent_text[i]])

    # permet de ne pas conserver les phrases en double sans alterer l'ordre des phrases
    resume_article = list(dict.fromkeys(resume_article))
    text_resume = ' '.join(resume_article)

    return text_resume

# fonction de coloration de mots en couleur : bleu (pas de signification particulière)

def blue_color_func(word, font_size, position, orientation, random_state=None,
                **kwargs):
    # return "hsl(243,74%%, %d%%)" % random.randint(38, 63)   # hsl(243, 74%, 43%) faire la modif pour le bleu
    return "hsl(228,87%%, %d%%)" % random.randint(10, 45)

# création d'un nuage de mot sur l'ensmble de l'article

def wordcloud_article(article):
    """
    Looking for pattern, entities, and most commons words, in a given text, to create a list of words that can be used to make a wordcloud (with clean and wordcloud_resume_list functions).
    Plot a wordcloud (blue) of words/expressions in a text.
    Using Spacy, nltk and WordCloud.

    ! Warning ! : you first need to run blue_color_func, clean and wordcloud_resume_list (in sentiments_resume) to use this function.

    input : article(string) = text to analyse.
    output : plot of a wordcloud (blue)
    """
    # condition de securite si l'article est vide cela ne bloquera pas le processus
    if article == '':
        list_cloud=['...']
    else:
        list_cloud = wordcloud_resume_list(article, 100)

    word_cloud_dict = Counter(list_cloud)

    wordcloud_resume = WordCloud(background_color = 'white', collocations = False, max_font_size = 200, 
                                 min_font_size = 10).generate_from_frequencies(word_cloud_dict)

    plt.figure(figsize=(12,25))
    plt.imshow(wordcloud_resume.recolor(color_func = blue_color_func),
            interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    
    return plt.show()


# creation contenu global : rassemble tous les articles en un seul texte à analyser
    
def global_text(df, column='Content') :
    """ 
    Join several rows of strings in a dataframe into a single string.
    Clean them from stopwords and numeric datas.

    ! Warning ! : you first need to run clean (in sentiments_resume) to use this function.

    input : df(dataframe) = dataframe in which are the strings to join
            column(string) = name of the column in which are the strings to join
    output : string (all strings joined into a single one)
    """
    all_text = ' '.join(df[column])
    text = clean(all_text)
    text2 = ' '.join(text)
    return text2

# creation d'un visuel avec deux nuages de mots : un avec les mots positifs (vert), un avec les mots negatifs (rouge) concernant
# l'ensemble des articles 

def wordcloud_global_sentiments(text_global, keyword, keyword2):
    """
    Analyse sentiment of each word of a string, and display most frenquently used words into two wordclouds : one for positive words (green), one for negative words (red).
    Predicting sentiment with Bert and Hugging Face, using WordCloud to plot wordclouds.

    ! Warning ! : you first need to run green_color_func, red_color_func and clean (in sentiments_resume) to use this function.

    input : text_global(string) = text to analyse, here : content of all the press articles into a single string.
            keyword(string) = keyword used for the search of the press articles, to remove from wordclouds.
            keyword2(string) = keyword used for the search of the press articles, to remove from wordclouds.
    output : plot with two wordclouds (green : positive words, red : negative words)
    """
    # decoupage du texte global en mots
    word_text = nltk.word_tokenize(text_global)
    list_neg = []
    list_pos = []
    # analyse de chaque mot 
    for word in word_text:
        score = multilang_classifier([word])
    # a chaque prediction de sentiment un accuracy score est associe
    # on ne prend en compte les predictions que si elles ont un score superieur ou egal a 0.4
        if score[0]['score'] >= 0.4:
            if (score[0]['label'] in ['5 stars','4 stars']):
                list_pos.append(word)

            elif score[0]['label'] in ['1 star','2 stars']:
                list_neg.append(word)

    # transformation des listes en texte puis wordcloud : 

    # pour le postitif : 
    all_text_pos = ' '.join(list_pos)
    text_pos = clean(all_text_pos)
    text2_pos = ' '.join(text_pos)

    wordcloud_global_pos = WordCloud(background_color = 'white', stopwords = [keyword,keyword2], 
                             collocations = True, width=480, height=480, max_font_size=200, min_font_size=10
                             ).generate_from_text(text2_pos)

    #pour le negatif :
    all_text_neg = ' '.join(list_neg)
    text_neg = clean(all_text_neg)
    text2_neg = ' '.join(text_neg)

    wordcloud_global_neg = WordCloud(background_color = 'white', stopwords = [keyword,keyword2], 
                             collocations = True, width=480, height=480, max_font_size=200, min_font_size=10
                             ).generate_from_text(text2_neg)

    # plot avec deux wordcloud : 

    fig, ax = plt.subplots(figsize = (20,10))

    # Définition du premier graphique
    ax1 = plt.subplot(121) 
    ax1.imshow(wordcloud_global_pos.recolor(color_func=green_color_func),
            interpolation="bilinear")
    ax1.axis("off")
    ax1.margins(x=0, y=0)

    # Définition du second graphique
    ax2 = plt.subplot(122) 
    ax2.imshow(wordcloud_global_neg.recolor(color_func=red_color_func),
            interpolation="bilinear")
    ax2.axis("off")
    ax2.margins(x=0, y=0)

    return plt.show()

# creation d'un nuage de mots concernant l'ensemble des articles

def global_wordcloud(text):
    """
    Wordcloud of the most frequently used words in a text (blue).
    Tokenizing text into words with nltk, using WordCloud to plot wordclouds.

    ! Warning ! : you first need to run blue_color_func (in sentiments_resume) to use this function.

    input : text(str) : text from stopwords to analyse, 
    in The press Watch : content of all the press articles into a single string, 
    that have been unified and cleaned with global_text function.
    output : plot with a wordcloud (blue)
    """
    wordcloud_global = WordCloud(background_color = 'white', width=480, height=480, 
                                max_font_size=200, min_font_size=10).generate_from_text(text)
    
    plt.figure(figsize=(12,25))
    plt.imshow(wordcloud_global.recolor(color_func = blue_color_func), 
            interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)

    return plt.show()