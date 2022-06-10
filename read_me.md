# About the project 

The Press Watch aims to follow publication of news online about a theme or keywords in particular.
It can get you quick access to : 
- medias which have published the articles
- what is positive, what is negative in these articles
- what is the general sentiment of this articles
- what is said about this theme or keywords in these articles

1. Seek press articles online

In news_scraping : url_seeker(keyword, keyword2, nombre_resultat = 5)
    """
    Seek press articles online with GoogleNews python library.

    input : keyword(str) = most important keyword to look for in press articles
            keyword2(str) = second keyword to add to your search in order to get more pertinence
            nombre_resultat(int) = number of press articles you want to look for (chronogical order from recent to past)
    output : Dataframe (we will use title, media, date, link)
    """

2. Look for content of these press articles

In news_scraping : AutoScrape(link)
    """
    Scrape content of online press articles.

    input : link(str) = url link of the website on which you want to scrape the content of a press     article
    output : string (of the content of the press article, or "Contenu non récupéré" if content couldn't be found)
    """    

3. Analyse sentiments of each article

- by defining positive, negative and neutral sentences

In sentiments_resume : sentiment_analysis(df_test, column='Content')
    """
    Define positive, negative and neutral sentences in a text of less than 512 caracters.
    Tokenizing in sentences with nltk, predicting sentiment with Bert and Hugging Face.

    input : df_test(dataframe) = dataframe in which is store the text to analyse
            column(string) = name of the column in which is store the text to analyse
    output : dataframe (with 3 new columns added, each containing positive sentences, negative sentences, neutral sentences)
    """

- by using them to define general sentiment of the article
In sentiments_resume : final_sentiment(df_test)
    """
    Define global sentiment of a text using prior results from function sentiment_analyses. 
    Which means that the text has already been divided in sentences dispatched by positive, negative, and neutral feeling.
    
    input : df_test(dataframe) = dataframe in which positive, negative and neutral sentences are stored respectively in columns named 'stars_pos', 'stars_neg', 'stars_neut'.
    output : dataframe (with a new column added, contaning final sentiment ('positif' for positive, 'negatif' for negative, 'neutre' for neutral, or 'non renseigné' is sentiment not found))
    """

4. Make a resume of the article around the chosen theme/keywords
In sentiment_resume : resume(text, keyword)
    """
    Make a resume of the text arount the chosen them/keywords.

    Tokenizing text in sentences, loop on all sentences of the text and : 
    return the sentence just before a sentence containing the keyword, the sentence containing the keyword, and the sentence just after a sentence containing the keyword.
    Remove duplicates and keep chronogical order of the reading of the text.

    input : text(str) = text for which you need a resume
            keyword(str) = keyword to look for in sentences.
    output : string (resume of the text)
    """

5. Give general feedbacks about the search : 

- simple statistics : 

Using plotly and results from previous functions.

number of articles by media

number of articles by sentiment

number of articles by publication date and sentiment

- negative and positive words used
In sentiments_resume : wordcloud_global_sentiments(text_global, keyword, keyword2)
    """
    Analyse sentiment of each word of a string, and display most frenquently used words into two wordclouds : one for positive words (green), one for negative words (red).
    Tokenizing text into words with nltk, predicting sentiment with Bert and Hugging Face, using WordCloud to plot wordclouds.

    ! Warning ! : you first need to run green_color_func and red_color_func (in sentiments_resume) to use this function.

    input : text_global(string) = text to analyse, here : content of all the press articles into a single string.
            keyword(string) = keyword used for the search of the press articles, to remove from wordclouds.
            keyword2(string) = keyword used for the search of the press articles, to remove from wordclouds.
    output : plot with two wordclouds (green : positive words, red : negative words)
    """

- most frequently words used 
In sentiments_resume : global_wordcloud(text)
    """
    Wordcloud of the most frequently used words in a text (blue).
    Tokenizing text into words with nltk, using WordCloud to plot wordclouds.

    ! Warning ! : you first need to run blue_color_func (in sentiments_resume) to use this function.

    input : text(str) : text from stopwords to analyse, 
    in The press Watch : content of all the press articles into a single string, 
    that have been unified and cleaned with global_text function.
    output : plot with a wordcloud (blue)
    """

6. Prepare text for wordclouds

- Clean text 
In sentiments_resume : clean(texte)
    """ 
    Tokenize a text and remove stopwords and numeric data.

    input : texte(str) = text to clean.
    output : string (cleaned text)
    """

- Define a list of words/expressions to return
In sentiments_resume : wordcloud_resume_list(article, nb_most_commons = 30)
    """
    Looking for pattern, entities, and most commons words, in a given text, to create a list of words that can be used to make a wordcloud.
    Using Spacy and nltk.

    input : article(str) = text to analyse.
            nm_most_communs(int) = number of most commons words to keep in our list.
    output : list of words/expressions.
    """

7. Give individual feedbacks about the search (article by article): 

- informations (title, date of publication, url link)

Using results from previous functions.

- negative and positive words/expressions used

In sentiments_resume : wordclouds_stars(df, row, keyword, keyword2)
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

- sentiment of the article

Using (3.) In sentiments_resume : final_sentiment(df_test)
Displaying sentiment with an image related to it : 
positive : green smiling smiley ('img_pos.png')
negative : red not mouth down smiley ('img_neg.png')
neutral : orange mouth as a line smiley ('img_neutre.png')

Pictures are stored in 'packages' folder.

- keywords / key expresssions used in this article
In sentiments_resume : wordcloud_article(article)
    """
    Looking for pattern, entities, and most commons words, in a given text, to create a list of words that can be used to make a wordcloud (with clean and wordcloud_resume_list functions).
    Plot a wordcloud (blue) of words/expressions in a text.
    Using Spacy, nltk and WordCloud.

    ! Warning ! : you first need to run blue_color_func, clean and wordcloud_resume_list (in sentiments_resume) to use this function.

    input : article(string) = text to analyse.
    output : plot of a wordcloud (blue)
    """

- sentences that mention the chosen theme/keywords
Using (4.) In sentiment_resume : resume(text, keyword)
Displaying the text.

# Requirements

beautifulsoup4==4.11.1
GoogleNews==1.6.3
matplotlib==3.5.1
nltk==3.7
numpy==1.21.5
pandas==1.0.5
Pillow==9.1.1
plotly==5.8.0
requests==2.27.1
spacy==3.3.0
streamlit==1.8.1
torch==1.11.0
transformers==4.19.2
wordcloud==1.8.1
tensorflow==2.8.0

# ways to improve tool

- find sentiment analysis models based on newspaper article and not comments
- scrap from bing which has less shields than google news
- specialized this tool to one theme
