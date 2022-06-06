# pip install --upgrade GoogleNews

from GoogleNews import GoogleNews
import pandas as pd

import requests
import json
from bs4 import BeautifulSoup
import time
import numpy as np

# recherche d'articles de presse en ligne

def url_seeker(keyword, keyword2, nombre_resultat = 5) :
    """
    Seek press articles online with GoogleNews python library.

    input : keyword(str) = most important keyword to look for in press articles
            keyword2(str) = second keyword to add to your search in order to get more pertinence
            nombre_resultat(int) = number of press articles you want to look for (chronogical order from recent to past)
    output : Dataframe (we will use title, media, date, link)
    """
    if nombre_resultat <= 20 :  # chaque page retourne 10 articles
        nb_page = 3
    elif nombre_resultat <= 30 :
        nb_page = 4
    elif nombre_resultat <= 50 :
        nb_page = 6
    elif nombre_resultat <= 100 : 
        nb_page = 11

    # mise en place de l'outil de recherche GoogleNews en français
    googlenews = GoogleNews(lang='fr', region='FR') 
    googlenews.clear()
    # lancement de la recherche avec les mots clés choisis
    googlenews.search(keyword+' '+keyword2)

    # création d'un boucle pour récupérer tous les résultats pour les paramètres choisis :
    list_result = []
    for i in range(1 , nb_page) : 
            result_test = googlenews.page_at(i)
            if result_test != []:
                for article in result_test:
                    list_result.append(article)
    # on place les résultats dans un dataframe
    df_search = pd.DataFrame(list_result)
    # on affiche uniquement le nombre de résultats souhaité ou bien l'intégralité
    # du dataframe s'il y a moins de résultats que souhaité
    if df_search.shape[0] >= nombre_resultat:
        df_search = df_search.head(nombre_resultat)

    return df_search


# suppression d'articles non pertinents à partir d'un mot clé contenu dans le titre d'un article
# (titre présent dans le dataframe de sortie des résultats de recherche GoogleNews)
def pertinence_seeker(df_search, keyword_to_drop):
    df_search = df_search[~df_search['title'].str.contains(keyword_to_drop, case = False)].reset_index(drop=True) 
    return df_search


# recuperation du contenu des articles de chaque lien url recupere lors de la recherche GoogleNews

def AutoScrape(link):   
    """
    Scrape content of online press articles.

    input : link(str) = url link of the website on which you want to scrape the content of a press     article
    output : string (of the content of the press article, or "Contenu non récupéré" if content couldn't be found)  
    """    
    # par defaut le contenu sera signale comme non recupere
    Content = 'contenu non récupéré'
    url = link

    # évite problème en cas de scraping de nombreux articles
    time.sleep(1)

    # tentive de connection via un try - except pour ne pas bloquer le processus en cas 
    # de problème de réponse
    try :
        navigator = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'
        html = requests.get(url, headers={'User-Agent': navigator})
        
    except ConnectionError:
        Content = 'ConnectionError : contenu non récupéré'

    except :
        Content = 'Error : contenu non récupéré'

    # récupération de la soupe du site scrapé 

    soup = BeautifulSoup(html.content, 'html.parser')

    if len(soup)>0 :
        content = soup.find_all()

        # on cherche à récupérer tous les titres, sous titres et balises p
        light_soup = content[0].find_all(name=['h1','h2','p'])

        # on crée une liste contenant les balises et les attributs y étant associés
        list_name = []
        for row in content:
            cells = row.find_all(['h1','h2','p'])
            for cell in cells:
                list_name.append(cell.name)

        # on recupere le texte de chaque balise que l'on place dans une liste
        list_text_article = []

        for i in range(0,len(light_soup)) : 
            if list_name[i] in ['h1','h2']:
                # on peut recuperer le texte de toutes les balises h1 et h2
                # qui correspondent aux titres et sous titres sur un site 
                title = light_soup[i].text
                # si le titre est suivi d'un espace on supprime cet espace
                if len(title)>=1 and title[-1] == ' ':
                    text = title[:-1] + '.'
                else :
                    text = title + '.'
                # le text est ajoute a notre liste
                list_text_article.append(text)

            elif (list_name[i] == 'p') and (len(light_soup[i].text) >= 185):
                # si le texte est contenu dans une balise p et est supérieur à 185 caractères
                # on peut considérer qu'il s'agit d'un paragraphe et le récupérer
                text = light_soup[i].text
                # on verifie que le paragraphe ne soit pas un paragraphe 
                # demandant d'accepter les cookies puis on l'ajoute à notre liste
                if not ('Cookies' in text or 'cookies' in text) :
                    list_text_article.append(text)
        
        text_complet = ' '.join(list_text_article)
        if len(text_complet) > 5000 :
            Content = text_complet[0,5000]
        else :     
            Content = text_complet
    else : 
        Content = 'contenu non récupéré'
    # par securite si malgre toutes ses etapes le contenu est vide on fait 
    # en sorte de le signaler :
    if Content == '':
        Content = 'contenu non récupéré'

    return Content