import streamlit as st
st.set_page_config(page_title='Crowd Perspectives',  layout='wide', page_icon=':chart_with_upwards_trend:',menu_items=None)
import pandas as pd
import numpy as np
import plotly.offline as po
import plotly.graph_objs as pg
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk import FreqDist
import nltk
from nltk.corpus import stopwords
import re
import base64
import requests
from IPython.display import display
from IPython.display import HTML
import streamlit.components.v1 as components
import gdown
from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Plus
import os
#nltk.download('punkt')
#nltk.download('stopwords')

############### Load Dataset ########################
#@st.cache(hash_funcs={"_thread.RLock": lambda _: None})
def load_data():
    # # path 
    # path = './Data' 
    # # Create the directory 
    # try: 
    #     os.mkdir(path) 
    # except OSError as error: 
    #     print(error)
    # print("Directory '% s' created" % path)
 
    # ##Download the dataset
    # try:
    #   url = st.secrets["data_url"]
    # except:
    #   url = "https://drive.google.com/drive/folders/1YuoS-eTbGpMOPoJiA-rtOZGWxiGE5qDb"
    # gdown.download_folder(url,quiet=True,use_cookies=True)
    
    # importing the zipfile module
  from zipfile import ZipFile
  import os
  # loading the temp.zip and creating a zip object
  with ZipFile(os.getcwd()+'/'+'Data.zip', 'r') as zObject:

    # Extracting all the members of the zip
    # into a specific location.
    zObject.extractall(
      path=os.getcwd())
    
    
    return "Done Load"
##############################
load_data()
###########################
def show_tweet(link):
    '''Display the contents of a tweet. '''
    url = 'https://publish.twitter.com/oembed?url=%s' % link
    response = requests.get(url)
    html = response.json()["html"]
    return html
########## search_tweets ##############
def search_tweets(quer,dfs):
  all_data = pd.concat(dfs, ignore_index=True)
  tokenized_corpus = [doc.split(" ") for doc in all_data["clean_text"]]
  bm25 = BM25Plus(tokenized_corpus)
  tokenized_query = quer.split(" ")
  results=final_new_menu = list(dict.fromkeys(bm25.get_top_n(tokenized_query, all_data["clean_text"],n=150)))
  df_serch=pd.DataFrame()
  for i in range(len(results)):
    df_serch=df_serch.append(all_data.loc[all_data['clean_text'] == results[i]])
  return df_serch 
###############################
#st.image('header.jpg', use_column_width=True)
data_di={"Cars":["Choose One","KIA","B.M.W","Mercedes Bens","Hyundai","Peugeot","Search in All"],
         "Companies":["Choose One","Microsoft","Vodafone","Amazon","Huawei","Dell","Samsung"],
         "Celebrities":["Choose One","Paris Hilton","Elon Musk","Mark Zuckerberg","Angelina Joly","Brad bet","Adele"],
         "Sport":["Choose One","Messi","Cristiano Ronaldo","Football","Fitness","Summing","Kickboxing"],
         "Recipes":["Choose One","Pizza","Koshare","Biryani","Sushi","Burger","Shawarma"],
         "Carton":["Choose One","Case Closed","Tom and Jerry","Superman","Batman","Cinderella","Ninja Turtles"],
         "Places":["Choose One","Pyramids","Khalifa Tower","Evil Tower","Leaning Tower","Statue of Liberty","Taj Mahal"]
        }
k=[]
for i,j in enumerate(data_di):
  k.append(j)
t1,t2,t3,t4 = st.columns([0.25,1,1,1])


t1.image('Picture.png', width = 100)

t2.title("**Crowd Perspectives **")

with t3:
  top_select = st.selectbox('', k)
  
with t4:
  com_select = st.selectbox('', data_di[top_select])  
##############  colors####################3
plat=["#334553","#0cbce4","#5baee5","#0c819c","#703770"]

############# Background Image ################
main_bg = "Wave-Background-Graphics-1-1-580x387.png"
main_bg_ext = "png"

side_bg = "Picture.png"
side_bg_ext = "png"
#.reportview-container
st.markdown(
    f"""
    <style>
    .stApp {{
       background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
       background-size:cover;
       
      
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>""",unsafe_allow_html=True)
#################################################33
#################################################
data = dict(type = 'choropleth',
        locations = ["Egypt","Brazil","UAE","USA","Germany"],
        locationmode = 'country names',
        #locations = ['AL', 'AK', 'AR', 'CA'],
        #locationmode = 'USA-states',
        z = [1,2,30,40,50],
        text = ['alabama', 'alaska', 'arizona', 'pugger', 'california'])

####################################################



def visi (df,df_age,df_hash,df_em):

  layout = pg.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    coloraxis_showscale=False,

    )

  
########################Clean the data#######################3
##
##  # removing short words/ punctuation
##  df["tweet"]= df["tweet"].apply(lambda x: " ".join ([w for w in x.split() if len (w)>3]))
##  for i in range (len(df["tweet"])):
##    df["tweet"][i] = re.sub(r"(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+", "", df["tweet"][i])
##  # tokenize the data
##  def tokenize(text):
##      tokens = re.split("\W+", text)
##      return tokens
##  df["tweet"]= df["tweet"].apply (lambda x: tokenize(x.lower()))
##  # remove stopwords
##  stop_words=set(stopwords.words("english"))
##  def remove_stopword(text):
##      text_nostopword= [char for char in text if char not in stop_words]
##      return text_nostopword
##
##  df["tweet"]= df["tweet"].apply(lambda x: remove_stopword(x))
##  # stemming
##  ps = nltk.stem.porter.PorterStemmer()
##  def stem(tweet_no_stopword):
##     text = [ps.stem ( word) for word in tweet_no_stopword]
##     return text
##
##  df["tweet"]= df["tweet"].apply(lambda x: stem(x))
######################################################
##
##  hassh=[]
##  final_ha=[]
##  symbols = ['.', '£', '-', '!', '(', ')', ':',"'",']','[']
##  for i in range(len(df["hashtags"])):
##    if df["hashtags"][i] != '[]':
##      for j in symbols:
##        df["hashtags"][i]=df["hashtags"][i].replace(j,'')
##      hassh.append(df["hashtags"][i].split(","))
##  for i in range(len(hassh)):
##    for j in range(len(hassh[i])):
##      final_ha.append(hassh[i][j])

################################ First Column################################################
  em1,chart4, chart5,chart6,chart7 = st.columns([0.25,1,1,1,1])

################# Most words ####################
    
  with chart4:

    symbols = ['.', '£', '-', '!', '(', ')', ':', ',',"'",']','[']

    l=""
    #for i in range(len(df["clean_text"])):
    for i in df["clean_text"]:
      #l= l + str(df["clean_text"][i])
      l= l + i
    for i in symbols:
      l=l.replace(i,'')

    split_it = l.split()

    freq_dist_pos = FreqDist(split_it).most_common(10)
    item=[]
    values=[]

    for i,j in freq_dist_pos:
      item.append(i)
      values.append(j)
    with st.expander("Most Words", True):
      st.write(px.bar(x=values,y= item,text=values,orientation='h',color=item
                      ,color_discrete_sequence=plat,labels={"x": "","y": ""},
                    width=400,title="Most Words").update_layout(layout).update_traces(showlegend=False))

##################### Sensment analysis ###########################33

  with chart5:
    from textblob import TextBlob
    def get_tweet_sentiment(tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(tweet)
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

##    ss=[]
##    for i in range(len(df["tweet"])):
##      ss.append(get_tweet_sentiment(str(df["tweet"].iloc[i])))

    Counter_sen = FreqDist(df["sentiment"])
    most_sen = Counter_sen.most_common(len(df["sentiment"].unique()))
    item_s = []
    values_s = []
    for i,j in most_sen:
      item_s.append(i)
      values_s.append(j)
    with st.expander("Sentment Analysis", True):
      st.write(px.bar(x=values_s,y= item_s,text=values_s,orientation='h',color=item_s,
                      color_discrete_sequence=plat,labels={"x": "","y": ""},
                    width=400,title="Sentment Analysis").update_traces(showlegend=False).update_layout(layout))

##################### Most Hashtages ###################

  with chart6:
    with st.expander("Most Hashtages", True):
       
      #freq_hash = FreqDist(df["1.png"]).most_common(10)
      item_h=df_hash["hashtag"].iloc[0:10]
      values_h=df_hash["counts"].iloc[0:10]

##      for i,j in freq_hash:
##        item_h.append(i)
##        values_h.append(j)

      st.write(px.bar(x=values_h,y= item_h,text=values_h,orientation='h',color=item_h,
                      color_discrete_sequence=plat,labels={"x": "","y": ""},
                width=400,title="Most Hashtages").update_traces(showlegend=False).update_layout(layout))
#######################################################
  
  with em1:
        try:
          st.image('1.png', width = 70)
          st.markdown(f"<h1 style='text-align: center;color: green;'>{round((values_s[item_s.index('positive')]/sum(values_s))*100)}%</h1>", unsafe_allow_html=True)
        except:
          
          st.markdown(f"<h1 style='text-align: center;color: green;'>{round(0*100)}%</h1>", unsafe_allow_html=True)
        try:
          st.image('2.png', width = 70)
          st.markdown(f"<h1 style='text-align: center;'>{round((values_s[item_s.index('neutral')]/sum(values_s))*100)}%</h1>", unsafe_allow_html=True)
        except:
          st.markdown(f"<h1 style='text-align: center;color: green;'>{round(0*100)}%</h1>", unsafe_allow_html=True)
        try:
          st.image('3.png', width = 70)
          st.markdown(f"<h1 style='text-align: center; color: red;'>{round((values_s[item_s.index('negative')]/sum(values_s))*100)}%</h1>", unsafe_allow_html=True)
        except:
          
          st.markdown(f"<h1 style='text-align: center;color: green;'>{round(0*100)}%</h1>", unsafe_allow_html=True)
        st.markdown("***")
##############Word Cloud##################33

  with chart7:
    from wordcloud import WordCloud, ImageColorGenerator
    from PIL import Image
    import urllib
    import requests
    # combining the image with the dataset
    long_string = ','.join(list(df['clean_text'].values))
    Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream = True).raw))

    # We use the ImageColorGenerator library from Wordcloud 
    # Here we take the color of the image and impose it over our wordcloud
    image_colors = ImageColorGenerator(Mask)

    # Now we use the WordCloud function from the wordcloud library 
    wc = WordCloud(background_color="white",contour_width=3, contour_color='#6DB1E0', max_words=80,collocations=False, mask = Mask).generate(long_string)

    # Size of the image generated 
    fig = plt.figure(figsize = (10, 20))

    # Here we recolor the words from the dataset to the image's color
    # recolor just recolors the default colors to the image's blue color
    # interpolation is used to smooth the image generated 
    plt.imshow(wc.recolor(color_func = image_colors), interpolation = "hamming")
    plt.axis('off')
    
    with st.expander("Wrod Cloud", True):
      st.pyplot(fig)



#################################### Second Column ########################################
  retweet,chart1, chart2,chart3 = st.columns([0.30,1,1,1])

  with retweet:
    st.markdown("**Highest  No. of Retweet**")
    number1 = int(max(df["retweet_count"]))
    st.markdown(f"<h1 style='text-align: center;'>{number1}</h1>", unsafe_allow_html=True)
    st.markdown("***")
    st.markdown("**Highest  No. of Likes**")
    number2 = int(max(df["like_count"]))
    st.markdown(f"<h1 style='text-align: center;'>{number2}</h1>", unsafe_allow_html=True)
    st.markdown("***")
    st.markdown("**No. of Hashtags**")
    #number3 = len(np.unique(np.array(final_ha)))
    number3 = len(df_hash["hashtag"])
    st.markdown(f"<h1 style='text-align: center;'>{number3}</h1>", unsafe_allow_html=True)


  with chart1:
##    chart_data = df['created_at']
##    for i in range(len(df['created_at'])):
##      chart_data[i]=chart_data[i][0:chart_data[i].index("T")]
    with st.expander("Tweetes Per Time", True):
      st.write(px.line(x =df['created_at'].unique(),y=df['created_at'].value_counts(),width=450,
                       color_discrete_sequence=plat,labels={"x": "","y": ""}).update_layout(layout))


  with chart2:
    chart_data =  pd.DataFrame(df_age['Age'].value_counts()[df_age['Age'].unique()],df_age['Age'].unique())
    with st.expander("Percentage of Age", True):
      st.write(px.bar(x =df_age['Age'].unique() ,y= df_age['Age'].value_counts()[df_age['Age'].unique()],text=df_age['Age'].value_counts()[df_age['Age'].unique()],orientation='v',
                    color=df_age['Age'].unique(),color_discrete_sequence=plat,
                      width=400,labels={"x": "","y": ""},title="").update_layout(layout).update_traces(showlegend=False))

  with chart3:
    chart_data =  pd.DataFrame(df_age['Gender'].value_counts()[df_age['Gender'].unique()],df_age['Gender'].unique())
    with st.expander("Percentage of Gender", True):
      st.write(go.Figure(data=[go.Pie(labels=df_age['Gender'].unique(),
                                    values=df_age['Gender'].value_counts()[df_age['Gender'].unique()],
                                      hole=.3)]).update_traces(marker=dict(colors=plat)).update_layout(layout).update_layout(width=400,height=300,margin=dict(t=.1, b=0.1, l=0.1, r=0.1)))





#################################### Third Column ########################################
  side,chart8, chart9,chart10 = st.columns([0.30,1,1,1])
  def count_word(dataa,n):
      symbols = ['.', '£', '-', '!', '(', ')', ':', ',',"'",']','[']

      l=""
      for i in range(len(dataa)):
        l= l + str(dataa.iloc[i])
      for i in symbols:
        l=l.replace(i,'')

      split_it = l.split()

      freq_dist_pos = FreqDist(split_it).most_common(n)
      item=[]
      values=[]

      for i,j in freq_dist_pos:
        item.append(i)
        values.append(j)
      return item,values

  with chart8:
      pos_s = pd.DataFrame(df["clean_text"].loc[df["sentiment"] == "positive"])
      

      
      with st.expander("Most Positive Words", True):
          
          st.write(px.bar(x=count_word(pos_s["clean_text"],10)[1],y= count_word(pos_s["clean_text"],10)[0],text=count_word(pos_s["clean_text"],10)[1]
                          ,orientation='h',color=count_word(pos_s["clean_text"],10)[0]
                      ,color_discrete_sequence=plat,labels={"x": "","y": ""},
                    width=400,title="Most Positive Words").update_layout(layout).update_traces(showlegend=False))



  with chart9:
      neg_s = pd.DataFrame(df["clean_text"].loc[df["sentiment"] == "negative"])
      

      
      with st.expander("Most Negative Words", True):
          
          st.write(px.bar(x=count_word(neg_s["clean_text"],10)[1],y= count_word(neg_s["clean_text"],10)[0],text=count_word(neg_s["clean_text"],10)[1]
                          ,orientation='h',color=count_word(neg_s["clean_text"],10)[0]
                      ,color_discrete_sequence=plat,labels={"x": "","y": ""},
                    width=400,title="Most Negative Words").update_layout(layout).update_traces(showlegend=False))

  with chart10:
      Counter_em = FreqDist(df_em["Emotion"])
      most_em = Counter_em.most_common(6)
      item_em = []
      values_em = []
      for i,j in most_em:
            item_em.append(i)
            values_em.append(j)
      

      
      with st.expander("Emotion Analysis", True):
                st.write(go.Figure(data=[go.Pie(labels=item_em, values=values_em, textinfo='label+percent')]).update_traces(marker=dict(colors=plat)).update_layout(layout).update_layout(width=400,height=300,margin=dict(t=.1, b=0.1, l=0.1, r=0.1)))
       



###########Display Tweetes###############3

    # A list of the tweet urls, sorted by retweet count.
##  rt_links = df.sort_values(by= 'retweet_count', ascending = False)['link'].values
##
##  for url in rt_links[:5]:
##      print('🔥 ' * 19)
##      components.html(show_tweet(url),height=800)
  
 ## jas=df["entities"][0][df["entities"][0].index("expanded_url")+16:df["entities"][0].index("display_url")-12]
 ## components.html(show_tweet(jas), height=800)
######################
## Map
##  layout = dict(geo = dict(scope ='world',
##                           # bgcolor="black",
##                           countrycolor="black",
##                          #projection = {'type':"orthographic"},
##                              showocean= True,
##                              showlakes = True,
##                          lakecolor = 'rgb(0,191,255)'
##                          )
##                  )
##  x = pg.Figure(data = [data] , layout = layout)
##  st.write(x.update_layout(width=1000,coloraxis_showscale=False
##                           ,height=300
##                           ,margin=dict(t=.1, b=0.1, l=0.1, r=0.1)).update_traces(showscale=False))
################################################################################################################3

################################### Ngram of Data ###########################
##from nltk.util import ngrams
##symbols = ['.', '£', '-', '!', '(', ')', ':', ',',"'",']','[']
##
##l=""
##for i in range(len(neg_s["clean_text"])):
##  l= l + str(neg_s["clean_text"].iloc[i])
##for i in symbols:
##  l=l.replace(i,'')
##
##split_it = l.split()
##
##ll=list(ngrams(split_it,4))
##
##
##freq_dist_pos = FreqDist(ll).most_common(10)
##item=[]
##values=[]
##
##for i,j in freq_dist_pos:
##  item.append(i)
##  values.append(j)
#########################################
if com_select == "KIA":
  df=pd.read_csv("Data/Merged_KIA.csv")
  df_age=pd.read_csv("Data/KIA_age.csv")
  df_hash=pd.read_csv("Data/kia_popular_hashtags.csv")
  word_cloud="wordcloud/kia.jpeg"
  df_em = pd.read_csv("Data/emotion_data.csv")
  visi(df,df_age,df_hash,df_em)
elif com_select == "B.M.W":
  df=pd.read_csv("Data/Merged_BMW.csv")
  df_age=pd.read_csv("Data/BMW_age.csv")
  df_hash=pd.read_csv("Data/BMW_popular_hashtags.csv")
  word_cloud="wordcloud/BWM.jpeg"
  df_em = pd.read_csv("Data/emotion_data.csv")                                         
  visi(df,df_age,df_hash,df_em)
elif com_select == "Mercedes Bens":
  df=pd.read_csv("Data/Merged_Mercedes.csv")
  df_age=pd.read_csv("Data/Mersedis_age.csv")
  df_hash=pd.read_csv("Data/mercedes_popular_hashtags.csv")
  word_cloud="wordcloud/Mercede.jpeg"
  df_em = pd.read_csv("Data/emotion_data.csv")                                         
  visi(df,df_age,df_hash,df_em)
elif com_select == "Hyundai":
  df=pd.read_csv("Data/Merged_Hyundai.csv")
  df_age=pd.read_csv("Data/Hyundai_age.csv")
  df_hash=pd.read_csv("Data/hyundai_popular_hashtags.csv")
  word_cloud="wordcloud/hyaduia.jpeg"
  df_em = pd.read_csv("Data/emotion_data.csv")                                         
  visi(df,df_age,df_hash,df_em)
elif com_select == "Peugeot":
  df=pd.read_csv("Data/Merged_Peugeot.csv")
  df_age=pd.read_csv("Data/Peugeot_age.csv")
  df_hash=pd.read_csv("Data/peuog_popular_hashtags.csv")
  word_cloud="wordcloud/peugeot.jpeg"
  df_em = pd.read_csv("Data/emotion_data.csv")                                         
  visi(df,df_age,df_hash,df_em)
elif com_select == "Search in All":
  buff, col_tx, buff2 = st.columns([1,3,1])
  quer = col_tx.text_input('Search :')
  if quer  == "":
    st.markdown("")
  else:
  
    df = search_tweets(quer,[pd.read_csv("Data/Merged_KIA.csv"),pd.read_csv("Data/Merged_BMW.csv"),pd.read_csv("Data/Merged_Mercedes.csv")
                             ,pd.read_csv("Data/Merged_Hyundai.csv"),pd.read_csv("Data/Merged_Peugeot.csv")])
    #st.write(df.head())
    df_age=pd.read_csv("Data/Peugeot_age.csv")
    df_hash=pd.read_csv("Data/peuog_popular_hashtags.csv")
    word_cloud="wordcloud/peugeot.jpeg"
    df_em = pd.read_csv("Data/emotion_data.csv")                                         
    visi(df,df_age,df_hash,df_em)

else:
 
  st.markdown("")
