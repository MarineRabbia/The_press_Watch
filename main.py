## SCRIPT

from packages.news_scraping import *
from packages.sentiments_resume import *
from PIL import Image
import streamlit as st

# pip install plotly
import plotly.express as px


st.set_option('deprecation.showPyplotGlobalUse', False)

def definir_style():
    style_monapp = """
    .row_heading.level0 {display:none}
"""
    st.markdown(f"<style>{style_monapp}</style>", unsafe_allow_html=True)


def choix_page():
    pages = {
        'Ma veille de presse en ligne' : analyse_articles,
        "Qu'est ce que The press Watch ?": read_me,
        'Contact': a_propos
        }
    
    with st.sidebar:
        page = st.selectbox("Choisir une page :", list(pages.keys()))
    
    pages[page]()

def analyse_articles():
 
    st.title("The press Watch")
    st.subheader("Renseignez les éléments de votre recherche : ")
    # on peut proposer plus de 30 articles mais le délai de traitement serait trop long pour une demo
    nombre_resultat = st.slider("Choisir le nombre d'articles à rechercher", 0, 30, 5)    

    keyword = st.text_input('Entrer le premier mot clé associé à votre recherche : ')

    keyword2 = st.text_input('Entrer un second mot clé associé à votre recherche : ')

    if st.button('Lancer ma recherche  🚀 '):

        df_result = url_seeker(keyword, keyword2, nombre_resultat)

        df_result.drop(['img','datetime','desc'], axis = 1, inplace = True)

        st.write(df_result)

        # fonction à mettre en place si probleme de pertinence
        keyword_to_drop = ''

        if keyword_to_drop != '':
            df_result = pertinence_seeker(df_result, keyword_to_drop)

        # mise en place d'un placeholder pour faire patienter l'utilisateur durant le chargement
        placeholder = st.empty()
        
        placeholder.text(" 📰  Récupération des articles en cours...")

        df_result['Content'] = df_result['link'].apply(AutoScrape)

        placeholder.text(' 🤖  Analyse en cours...')

        df_result['Resume'] = df_result['Content'].apply(lambda x: resume(x, keyword))

        df_analysis = sentiment_analysis(df_result)

        df_final_sentiment = final_sentiment(df_analysis)

        pos = Image.open(r'packages/img_pos.png')

        neutre = Image.open(r'packages/img_neutre.png')

        neg = Image.open(r'packages/img_neg.png')

        # Analyse globale de l'ensemble des articles récupérés 

        st.header("Analyse générale")

        placeholder.text(" 📊  Préparation des visuels en cours ...")
   
        fig_media = px.histogram(df_final_sentiment, x = "media", 
                    title = "Nombre d'articles par media", 
                    color_discrete_sequence = px.colors.qualitative.Safe)
        fig_media.update_xaxes(range=[0, 10])
        
        st.plotly_chart(fig_media) 

        fig_sentiment = px.histogram(df_final_sentiment, x = "final_sentiment_hugging", 
                        color = "final_sentiment_hugging",
                        title = "Nombre d'articles par sentiment",
                        color_discrete_sequence = px.colors.qualitative.Safe,
                        labels = {"final_sentiment_hugging":"Sentiment"},
                        color_discrete_map = {
                "positif": "rgb(204,235,197)",
                "negatif": "rgb(251,180,174)",
                "neutre": "rgb(254,217,166)",
                "non renseigné": "rgb(222,203,218)"})
        fig_sentiment.update_xaxes(title_text = "Sentiment")
        st.plotly_chart(fig_sentiment) 

        fig_date = px.histogram(df_final_sentiment, x = "date", color = "final_sentiment_hugging",
                    labels = {"final_sentiment_hugging":"Sentiment"},
                    title = "Nombre d'articles par date de publication", 
                    color_discrete_sequence = px.colors.qualitative.Safe)
        fig_date.update_xaxes(title_text = "Date de publication")
        st.plotly_chart(fig_date)

        text_global = global_text(df_final_sentiment)

        placeholder.text(' 🦩  Création des nuages de mots...')

        wc_global_posneg = wordcloud_global_sentiments(text_global, keyword, keyword2)
        st.pyplot(wc_global_posneg)

        wc_global = global_wordcloud(text_global)
        st.pyplot(wc_global)

        placeholder.text(' 🦥  Affichage des analyses par article...')

        # Analyse article par article

        st.header("Analyse artice par article")

        for i in range(0,len(df_final_sentiment)):

            st.write(df_final_sentiment['title'][i])

            st.write(df_final_sentiment['date'][i])

            st.write(df_final_sentiment['link'][i])

            if 'contenu non récupéré' in df_final_sentiment['Content'][i]:
                st.write("Contenu de l'article non récupéré")
                
            else :
                st.write("Le positif et négatif de l'article :")
                wordclouds_stars(df_final_sentiment, i, keyword, keyword2)
                st.pyplot(wordclouds_stars(df_final_sentiment, i, keyword, keyword2))

                
                st.write("Le sentiment général de l'article :")
                if df_final_sentiment['final_sentiment_hugging'][i] == "positif":
                    st.image(pos, caption='positif')

                elif df_final_sentiment['final_sentiment_hugging'][i] == "negatif":
                    st.image(neg, caption='negatif')

                elif df_final_sentiment['final_sentiment_hugging'][i] == "neutre":
                    st.image(neutre, caption='neutre')


                st.write("Mots clés de l'article : ")

                if len(df_final_sentiment['Resume'][i]) > 400 :
                    wc_resume = wordcloud_article(df_final_sentiment['Resume'][i])
                    st.pyplot(wc_resume)
                else:
                    wc_content = wordcloud_article(df_final_sentiment['Content'][i])
                    st.pyplot(wc_content)

                st.write("Résumé de l'article :")
                st.write(df_final_sentiment['Resume'][i])

            st.write("_____________________________________")

        st.write(df_final_sentiment)
        placeholder.text(" 🏁  Merci pour votre patience, les résultats sont à présent complets.")

def read_me():
    st.header("The press Watch")

    st.markdown("**The press Watch** récupère de manière chronologique\
                les url d'articles de presse en ligne en fonction de mots clés recherchés.")
        
    st.markdown("Les sites sur lesquels sont diffusés les articles de presse contiennent de nombreuses \
                informations, ils sont tous construits d'une manière qui leur est propre, \
                l'objectif de **The press Watch** est de ne récupèrer que le contenu de l'article.")

    st.markdown("Grâce à cela, **The press Watch** indique ensuite quels sont les éléments positifs ou négatifs\
                de l'article, ainsi que son sentiment général. **The press Watch** vous permet d'accéder au résumé de l'article\
                sous forme de texte et de nuage de mots clés.")

    st.markdown("Enfin **The Press Watch** associe quelques indicateurs à votre recherche, permettant un aperçu sur les\
                médias publiant les articles de presse recherchés, sur le rythme de publication des articles de presse recherchés\
                et sur les tendances des sentiments dse articles de presse recherchés.")

def a_propos():
    col1, col2 = st.columns([1, 3])
    image = Image.open(r'packages/MarineRabbia.png')
    with col1:
        st.image(image)
    with col2:
        st.subheader("Marine Rabbia")
        st.markdown("Data Analyst")
        url_linkedin = "https://www.linkedin.com/in/marine-rabbia/"
        st.write("Plus d'informations sur [Linkedin](%s)" % url_linkedin)
        url_github = "https://github.com/MarineRabbia"
        st.write("Repository disponible sur [Github](%s)" % url_github)

    st.write("_____________________________________")

    col3, col4 = st.columns([1, 1])
    with col3 :
        st.subheader("Remerciements : ")
        url_wild = "https://www.linkedin.com/school/wild-code-school/mycompany/verification/"
        #url_yassine = 
        #url_celine = 
        st.write(" ⭐ [The Wild Code School](%s)" % url_wild)
        #st.write(" ⭐ [Yassine](%s)" % url_yassine)
        #st.write(" ⭐ [Céline](%s)" % url_celine)

    with col4:
        st.subheader("Avec la participation de : ")
        url_vincent = "https://www.linkedin.com/in/vincentfritsch/"
        url_sabrina = "https://www.linkedin.com/in/sabrina-hachouf-356a3b88/"
        url_guillaume = "https://www.linkedin.com/in/guillaume-bitaudeau-16280721b/"
        url_cyril = "https://www.linkedin.com/in/cyrilgamboa/"
        st.write("✨ [Vincent Fritsch](%s) - Data Analyst" % url_vincent)
        st.write("✨ [Sabrina Hachouf](%s) - Data Analyst" % url_sabrina)
        st.write("✨ [Guillaume Bitaudeau](%s) - Data Analyst" % url_guillaume)
        st.write("✨ [Cyril Gamboa](%s) - Data Analyst" % url_cyril)

if __name__ == "__main__":
    definir_style()
    choix_page()


