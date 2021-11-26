#from matplotlib.pyplot import title
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from modzy import ApiClient
from modzy._util import file_to_bytes
import json
from sklearn.manifold import TSNE
import numpy as np
from pyvis.network import Network
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.image('text.png')
#col1,col2 = st.columns([1,6])
st.image('subtext.png')
#df=pd.read_csv("https://raw.githubusercontent.com/pupimvictor/NetworkOfThrones/master/stormofswords.csv")
df = pd.read_csv("https://raw.githubusercontent.com/napoles-uach/Data/main/got1.csv")
df=df[['Source','Target','weight']]
#st.write(df)
#weigths=df['weight'].tolist()
def got_func():
  got_net = Network(height="600px", width="100%", heading='A song of Ice and Fire (Book 1) Graph')#,bgcolor='#222222', font_color='white')

# set the physics layout of the network
  #got_net.barnes_hut()
  got_net.force_atlas_2based()
  #got_net.show_buttons(filter_=True)
  #got_data = pd.read_csv("https://www.macalester.edu/~abeverid/data/stormofswords.csv")
  got_data = pd.read_csv("https://raw.githubusercontent.com/napoles-uach/Data/main/got1.csv")
  #got_data = pd.read_csv("stormofswords.csv")
  #got_data.rename(index={0: "Source", 1: "Target", 2: "Weight"}) 
  sources = got_data['Source']
  targets = got_data['Target']
  weights = got_data['weight']

  edge_data = zip(sources, targets, weights)

  for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]

    got_net.add_node(src, src, title=src, color='red')
    got_net.add_node(dst, dst, title=dst,color='red')
    got_net.add_edge(src, dst, value=w)

  neighbor_map = got_net.get_adj_list()

# add neighbor data to node hover data
  for node in got_net.nodes:
    node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
    node["value"] = len(neighbor_map[node["id"]])
 

  got_net.show("gameofthrones.html")

got_func()
HtmlFile = open("gameofthrones.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
#check_graph = st.sidebar.checkbox('Show Graph')
#if check_graph:
with st.expander('Show Graph'):       
    components.html(source_code, width=670,height=700)



text = open("edges.txt","w")
text.write('graph')
for i in range(len(df)):
  text.write('\n%s' % str(df.iloc[i][0]).replace(" ", "")+" "+str(df.iloc[i][1]).replace(" ", "")+" "+str(df.iloc[i][2]))
text.close()
f = open('edges.txt','r',encoding='utf-8')



client = ApiClient(base_url="https://app.modzy.com/api", api_key="zHXMIMi6uB9Ur0ldfngf.AVwEddcm4H4gEOcm5QtQ")
sources = {}

sources["my-input"] = {
    "edges.txt": f.read(),
}

@st.cache()
def res(sources):
    job = client.jobs.submit_text("sixvdaywy0", "0.0.1", sources)
    result = client.results.block_until_complete(job, timeout=None)
    return result

#job = client.jobs.submit_text("sixvdaywy0", "0.0.1", sources)


#result = client.results.block_until_complete(job, timeout=None)
result = res(sources)

#st.button('Download')
#st.balloons()
#st.stop()

results_json = result.get_first_outputs()['results.json']

x = results_json['Node Embeddings']

names_dict = []
vec_dict = []
for names in x:
  names_dict.append(names)

  v=x[names].split()
  vec_dict.append(v)

# convert a list of string numbers to a list of float numbers
def convert_to_float(l):
    return [float(i) for i in l]

vec_dict = [convert_to_float(i) for i in vec_dict]



chart_data=pd.DataFrame(vec_dict)
#traspose of the dataframe
chart_data=chart_data.T
#column names are the names of the nodes
chart_data.columns=names_dict



#st.bar_chart(chart_data['Aegon'])
#st.bar_chart(chart_data['Worm'])

@st.cache()
def do_tsne(vector,randomst,perp):
  tsne = TSNE(random_state=randomst,perplexity=perp)
  return tsne.fit_transform(vector)

digits_tsne = do_tsne(vec_dict,42,50)
#st.write('aqui')
#st.write(digits_tsne)
with st.sidebar.expander('TSNE'):
    clusters_n = st.slider('Number of clusters for TSNE', min_value=3, max_value=10, value=3)
kmeans = KMeans(n_clusters=clusters_n, random_state=0).fit(digits_tsne)

#st.write(kmeans.labels_)

ejex = []
ejey = []
indice=[]
for i in range(len( digits_tsne  )):
  ejex.append( digits_tsne[i][0] )
  ejey.append( digits_tsne[i][1] )
  indice.append(i)

dic = {'ejex':ejex,'ejey':ejey,'indice':indice}
df = pd.DataFrame(dic)
#add a column with the name of the node
df['nombre']=names_dict
df['labels']=kmeans.labels_

#st.write(df)

fig  = px.scatter(df,x='ejex',y='ejey',hover_data=['nombre'],color='labels')
#with st.sidebar.expander('TSNE'):
#check_tsne = st.sidebar.checkbox('Show TSNE plot')
#if check_tsne:
with st.expander('Modzi app 1'):
    '''
### Graph Embeddings
#### Description

This model can be used to explore possible relationships between entities, such as finding people who share similar interests or finding biological interactions between pairs of proteins. Graphs are particularly useful for describing relational entities, and graph embedding is an approach used to transform a graph’s structure into a format digestible by an AI model, whilst preserving the graph’s properties. 
Graph structures can widely vary in terms of their scale, specificity, and subject, making graph embedding a difficult task.
    '''

with st.expander('Graph Embedings'):
    st.write(chart_data)

with st.expander('Show TSNE plot'):
    st.plotly_chart(fig)

text = open("text.txt","r")
paragraph = text.read().split('\n\n')
paragraphs_df = pd.DataFrame(paragraph)
paragraphs_df.columns = ['paragraphs']
#selectbox = st.selectbox('Select a node', names_dict)
#list wit numbers from 0 to clusters_n
cluster_numbers = list(range(clusters_n))

@st.cache()
def res2(sources2):
    job = client.jobs.submit_text("a92fc413b5", "0.0.12", sources2)
    result = client.results.block_until_complete(job, timeout=None)
    return result

sel_cluster = st.sidebar.multiselect('Select clusters', cluster_numbers, default=cluster_numbers[:3])


lista_col=st.columns(len(sel_cluster))
check_df=st.sidebar.checkbox('Show cluster dataframe')
if check_df:
    ii=0


    for i in sel_cluster:
    
        with lista_col[ii]:
            st.write('Cluster '+str(i))
            st.write(df[df['labels']==i].nombre)
        ii=ii+1


with st.sidebar.expander('Select Node'):
    character=st.text_input('Enter a name','Eddard-Stark')
char1=character

        #replace - with space in character string
character=character.replace("-"," ")
        
block = paragraphs_df[paragraphs_df['paragraphs'].str.contains('==== '+character)]



block_str=block.paragraphs.tolist()
with st.sidebar.expander('Show paragraphs'):
    st.header(block_str[0])



sources2 = {}

sources2["my-input"] = {
    "input.txt": block_str[0],#str(block),
}





result2=res2(sources2)
results_json2 = result2.get_first_outputs()['results.json']

with st.expander('Modzi app 2'):
    '''
#### Named Entity Recognition, English
Description

Named entity recognition, also known as entity extraction, detects and classifies the entities in a piece of English text into four categories: 
persons, locations, organizations, and miscellaneous. The input to this model is an English text and the output is each word in the text labeled as one of these four categories or 'O' indicating that the word is not an entity. 
This model can add a wealth of semantic knowledge to any text content and helps with the understanding of the subject covered in any given text.
    '''
 
check=st.checkbox('Show results')
if check:

    cloud_list=[]
    for i in results_json2:
        if i[1]!='O':
            cloud_list.append(i[0])

#add entries in cloud_list to a single string
    cloud_str=''
    for i in cloud_list:
        cloud_str=cloud_str+i+', '
    wordcloud = WordCloud().generate(cloud_str)

# Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    st.pyplot()
    
    #st.bar_chart(chart_data[char1])

    with st.expander('Show tokens'):
        st.write(results_json2)
    #st.pyplot()

st.stop()
#print(f"job: {job}")


#with lista_col[0]:
#  st.write('Cluster')

#with lista_col[1]:
#  st.write('Cluster')
#with lista_col[2]:
#  st.write('Cluster')
#st.write(lista_col)
st.stop()
for i in sel_cluster:

    df_1=df[df['labels']==i]
    st.write(df_1['nombre'])

# evaluate the cosine similarity between two vectors in vec_dict
def cosine_similarity(v1, v2):
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 

# evaluate the distance between two vectors in vec_dict
def distance(v1, v2):
    dist = np.linalg.norm(np.array(v1) - np.array(v2))
    #take the gaussian distance
    return np.exp(-0.5*(dist/5)**2)
    #return np.linalg.norm(v1 - v2)

#st.write(np.array(vec_dict[0])- np.array(vec_dict[1]))
#st.stop()

#make a cosine symmetric matrix a plot with plotly express
def make_cosine_matrix(vec_dict):
    cosine_matrix = np.zeros((len(vec_dict), len(vec_dict)))
    for i in range(len(vec_dict)):
        for j in range(len(vec_dict)):
            cosine_matrix[i,j] = cosine_similarity(vec_dict[i], vec_dict[j])
    return cosine_matrix



# convert cosine_matrix to pandas dataframe
def cosine_matrix_to_df(cosine_matrix):
    df = pd.DataFrame(cosine_matrix)
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)
    return df

co_mat_df=cosine_matrix_to_df(make_cosine_matrix(vec_dict))
#st.write(co_mat_df)

#plot the cosine matrix wit plotly express
fig2 = px.imshow(co_mat_df,color_continuous_scale=px.colors.sequential.Plasma)

st.write(fig2)



st.stop()
#add names_dict as index to the dataframe
co_mat_df.index=names_dict
#co_mat_df.columns=names_dict
sumas=(co_mat_df.sum(axis=1))
sumas=sumas.sort_values(ascending=True)
st.bar_chart(sumas)

# make a vertical bar chart with plotly express for sumas dataframe in ascending order
#ig3 = px.bar(sumas.sort_values(ascending=False), x='index', y='sumas')
fig3 = px.bar(sumas, y=sumas.index, x=sumas.values, orientation='h')
st.write(fig3)
