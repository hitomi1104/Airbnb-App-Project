# import
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import warnings
import streamlit as st
from keras.layers import Dense, Activation
from keras.models import Sequential
from tensorflow import keras
from sklearn.cluster import KMeans
# folium
import folium
import folium.plugins as plugins
from streamlit_folium import folium_static

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# NLP
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import string


plt.style.use('ggplot')
alt.data_transformers.disable_max_rows()
# SETTING PAGE CONFIG TO WIDE MODE
# st.set_page_config(layout="wide")
# st.balloons()

# LAYING OUT THE TOP SECTION OF THE APP
row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    st.title("Airbnb in LA")
    st.image('airbnb.jpeg', width = 280, caption = "Where are you staying tonight?")
    'What makes the lists of Airbnb more expensive than the others 💭'

with row1_2:

    """
    Data Source: http://insideairbnb.com/get-the-data.html
    
    This App explores Airbnb data which someone webscraped from the actual website before 
    By sliding the slider on the left you can view different slices of price and explore different trends.
    """





# LOADING DATA

# data = pd.read_csv('/Users/hitomihoshino/Desktop/DS/Projects/projects/Airbnb_LA/listings.csv')
data = pd.read_csv('listings.csv')

st.header('Data Dictionary')
data_dict = {'id': 'Airbnbs unique identifier for the listing',
             'name': 'Name of the listing',
             'host_id': 'Airbnbs unique identifier for the host/user',
             'host_name': 'Name of the host. Usually just the first name(s).',
             'neighbourhood_group': 'The neighbourhood group as geocoded using the latitude and longitude against neighborhoods as defined by open or public digital shapefiles.',
             'neighbourhood': '',
             'latitude': 'Uses the World Geodetic System (WGS84) projection for latitude and longitude.',
             'longitude': 'Uses the World Geodetic System (WGS84) projection for latitude and longitude.',
             'room_type': '[Entire home/apt|Private room|Shared room|Hotel]',
             'price': 'daily price in local currency',
             'minimum_nights': 'minimum number of night stay for the listing (calendar rules may be different)',
             'number_of_reviews': 'The number of reviews the listing has',
             'last_review': 'The date of the last/newest review',
             'reviews_per_month': '',
             'calculated_host_listings_count': 'The number of listings the host has in the current scrape, in the city/region geography.',
             'availability_365': 'avaliability_x. The availability of the listing x days in the future as determined by the calendar. Note a listing may not be available because it has been booked by a guest or blocked by the host.',
             'number_of_reviews_ltm': 'The number of reviews the listing has (in the last 12 months)',
             'license': 'The licence/permit/registration number'
             }
data_dict = pd.DataFrame(data_dict, index=['descriptions']).T
st.table(data_dict)

'''
In the data cleaning processes, I droppped some of the variables such as id, which doens't give any insights in predicting the price of the house
I will explain those deatails in my code
'''

# DATA CLEANING
st.caption('( I am not showing Data Cleaning on my App but you can find it in my code )')
data.drop_duplicates(inplace=True)
# dropping meaningless columns
data.drop(['id', 'host_id', 'host_name'], axis=1, inplace=True)
# checking null values
data.isnull().sum(axis=0)
# dropping columns that are overlapping with other cols such as reviews, last_reviews, and reviews_per_month,
# and ones having too many nan values that make it difficult to fill with averages, forward-fill, or backward-fill
df = data.copy()
df.drop(["last_review", "reviews_per_month", "license"], axis=1, inplace=True)
#  Since there are only 2 nan values for the column name, fill nan with 'no info'
# df[df["name"].isnull()]
df = df.where(pd.notnull(df), "no info")
df.isnull().sum(axis=0)

# changing the name of columns
df.set_axis(['name', 'neighbourhood_group', 'neighbourhood', 'latitude', 'longitude',
             'room_type', 'price', 'minimum_nights', 'number_of_reviews',
             'number_of_host_listings', 'availability',
             'reviews_year'], axis=1, inplace=True)
# changing the order of the columns for the col price to be the first so the visualizations can look better and clearer
cols = df.columns.tolist()
cols = cols[6:12] + cols[0:6]
df = df[cols]

df.head()


################################################################################ EDA   ################################################################################
st.header('Exploratory Data Analysis')
st.subheader('Distributions and statistical analysis of the target variable Price')
'''
Exploring the target variable which is price and see its distributions
'''
row1_1, row1_2 = st.columns((1,1))

with row1_1:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_style(style='darkgrid')
    sns.distplot(df['price'], bins=100, kde=False, color='blue')
    st.pyplot(fig)
    '''
    As you can see some of the houses are very expensive which makes it outliers
    Those listings had outstanding features to be expensive. 
    '''
    st.text("")

    st.write('** Decision  **')
    '''
    Most of the listings are less than 500 dollers per night
    So, I will conisder the listings only less than 500 to do some further analysis and predicting the price
    '''

with row1_2:
    # dropping outliers; also considering even many 5 star hotels don't charge more than 500 for a normal stay
    df = df[df['price'] < 500]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_style(style='darkgrid')
    sns.distplot(df['price'], bins=100, kde=False, color='blue')
    st.pyplot(fig)
    '''
    **Statistics of the data (which I use to build ML models)**
    '''
    st.write(df['price'].describe(include=all))
    # Skew and kurtosis for SalePrice
    "Skewness: %f" % df['price'].skew()
    "Kurtosis: %f" % df['price'].kurt()
    "Mode: %f" % df['price'].mode()




##########################################################################################################################
st.subheader('Distributions and statistical analysis of all variables')
st.caption('Exclusing the colulmn name which I will show it in wordcloud instead later')
bar1, bar2 = st.columns((2,1))

with bar1:
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    df['minimum_nights'].value_counts().head(10).plot(ax=ax[0][0], kind='bar', title = 'Minimum night')
    df['number_of_reviews'].value_counts().head(10).plot(ax=ax[0][1], kind='bar', title = 'Number of reviews')
    df['number_of_host_listings'].value_counts().head(10).plot(ax=ax[0][2], kind='bar', title = 'Number of host listings')
    df['availability'].value_counts().head(10).plot(ax=ax[0][3], kind='bar', title = 'Availability')

    df['reviews_year'].value_counts().head(10).plot(ax=ax[1][0], kind='bar', title = 'Reviews per year')
    df['neighbourhood_group'].value_counts().head(10).plot(ax=ax[1][1], kind='bar', title = 'Neighborhood group')
    df['neighbourhood'].value_counts().head(20).plot(ax=ax[1][2], kind='bar',  title = "Neighborhood")
    df['latitude'].value_counts().head(10).plot(ax=ax[1][3], kind='bar', title = "Latitude")
    df['longitude'].value_counts().head(10).plot(ax=ax[2][0], kind='bar', title = 'Longitude')

    # fixing spcaing between bar charts avoiding overlapping words
    '''[reference] stackoverflow https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib'''
    fig.tight_layout()
    st.pyplot(fig)

    '''
    In this part, I made distribution plots for both numerical and categorical data and sorted values in descending order
    So, for instance in the variable neighborhood, I found "Venice" had the most listings.
    I was surprised to see that most of the listings had the restriction to stay more than 30 minimun nights! Also, most of the listings are
    listed for the first time but most of them are booked when I observe the number of host listings and availability variables. 
    '''
with bar2:
    # Showing all the unique values
    '''
    ** Unique values of each columns **
    '''
    df = df[(df['minimum_nights'] <= 31) & (df['room_type'] == 'Entire home/apt')]
    df.drop(columns='room_type', inplace=True)
    for col in df.columns:
        st.write(' - {} : {} unique values'.format(col, len(df[col].unique())))


##########################################################################################################################
st.subheader('Heatmaps')
'''[reference]: Altair documentation https://altair-viz.github.io/gallery/layered_heatmap_text.html '''

cor_data = (df.drop(columns=['name', 'neighbourhood_group', 'neighbourhood'])
            .corr().stack()
            .reset_index()
            .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal

base = alt.Chart(cor_data).encode(
    x='variable2:O',
    y='variable:O'
).properties(
    width=500,
    height=400
)

text = base.mark_text().encode(
    text='correlation_label',
    color=alt.condition(
        alt.datum.correlation > 0.5,
        alt.value('white'),
        alt.value('black')
    )
)



cor_plot = base.mark_rect().encode(
    color='correlation:Q'
)


cor_plot + text

cor1, cor2 = st.columns((3,2))
with cor1:
    # Top correlations
    corr_matrix = df.corr()
    d_top = corr_matrix['price'].sort_values(ascending=False)
    st.write('Top Correlations are: \n', d_top.head(10))
with cor2:
    # Bottom correlations
    d_down = corr_matrix['price'].sort_values(ascending=True)
    st.write('Top Negative Correlations are: \n', d_down.head(10))

################################################################# MAP #########################################################################
st.subheader('Exploring reginons with Maps')

map_1, map_2 = st.columns((2, 2))

with map_1:
    st.caption('Kmeans clustering of latitude and longitude')
    s = st.slider('select the number of iterations', 1, 20)
    kmeans = KMeans(n_clusters=5, max_iter=s, n_init=1)

    val = df[['longitude', 'latitude']]

    kmeans.fit(val)
    val["cluster"] = kmeans.predict(val)

    c = alt.Chart(val).mark_circle().encode(
        #     x ='longitude',
        #     y = 'latitude',
        x=alt.X('longitude', scale=alt.Scale(domain=[-119, -117])),
        y=alt.Y('latitude', scale=alt.Scale(domain=[33, 35])),
        color="cluster:N"
    )
    st.altair_chart(c, use_container_width=True)

with map_2:
    st.caption('folium map visualization')
    price_selected = st.slider("Select the range of the price", value=[0, 500])
    df_price = df[df['price'] <= price_selected[1]]
    df_price = df_price[df_price['price'] >= price_selected[0]]
    count = df_price[['latitude', 'longitude']]
    m = folium.Map([34, -118], zoom_start=7)


    plugins.Fullscreen(position='topright',  # Full screen
                       title='Click to Expand',
                       title_cancel='Click to Exit',
                       force_separate_button=True).add_to(m)


    plugins.MousePosition().add_to(m)  ## get coordinates.
    plugins.MarkerCluster(count).add_to(m)
    st.markdown("[reference] streamlit documentation of visualizing folium maps ")
    folium_static(m)
    st.markdown("[reference] https://python-visualization.github.io/folium/plugins.html")


################################################################################ NLP  ################################################################################
# s = st.slider("Select the range of the price", value=[0, 500])

# vectorize the word using tfidvectorizer from sklearn
st.subheader("Vectorizing the columns name and show the frequent words using WordCloud")
tf = TfidfVectorizer(stop_words='english', min_df=3)
tf.fit(df['name'])

name_tf = tf.transform(df['name'])
name_df = pd.DataFrame(name_tf.todense(), columns=tf.get_feature_names())

tf1, tf2, tf3 = st.columns((2,3,1))

with tf1:
    fig, ax = plt.subplots()
    top_texts = name_df.sum().sort_values(ascending=False)
    top_texts.head(15).plot(kind='barh')
    st.pyplot(fig)

with tf2:

    fig, ax = plt.subplots()
    # Create and generate a word cloud image:
    Cloud = WordCloud(width=500, height=400,
                      background_color='black',
                      stopwords=stopwords,
                      min_font_size=3,
                      min_word_length=0).generate_from_frequencies(top_texts)

    # background_color="white", max_words=50).generate_from_frequencies(top_texts)

    # Display the generated image:
    plt.imshow(Cloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)

with tf3:
    '''
    Words that describe the type of the listings such as parking and studio, comfort and convinience such as cozy, near, and modern were top words.
    
    In the future, I am interested to see the difference in words among different prices of the listings. 
    '''

################################################################################ ML  ################################################################################
st.header("Applying ML to predict the price of the listed Airbnb houses")
" ** Since I am dealing with a Regressor, I will use R^2 Scores to compare which models have the best score ** "

st.subheader("Part I: Prediction using only numeric columns")

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error as MSE
from pandas.api.types import is_numeric_dtype

numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

df_num = df[numeric_cols]

# Define target and predictors
X = df_num.copy()
y = X.pop('price')

# diving data into trainning and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10)

# standardlize the train/test sets of data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),
                       columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),
                      columns=X_train.columns)

# Applying Linear Regression
st.caption("LR")
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_cv = cross_val_score(lr, X_train, y_train, cv=5)

lr1, lr2, lr3 = st.columns((1, 2, 2))

with lr1:
    st.write('Cross-validated scores:', lr_cv.T)
    st.write('Average score:', lr_cv.mean())
    st.write('Trainning Score:', lr.score(X_train, y_train))
    st.write('Test Score:', lr.score(X_test, y_test))

with lr2:
    # collect the model coefficients in a dataframe
    df_coef = pd.DataFrame(lr.coef_, index=X_train.columns,
                           columns=['coefficients'])
    # calculate the absolute values of the coefficients

    df_coef['coef_abs'] = df_coef.coefficients.abs()

    coefs = pd.concat([df_coef['coefficients'].sort_values().head(4),
                       df_coef['coefficients'].sort_values().tail(3)])

    fig, ax = plt.subplots()
    coefs.plot(kind="barh", figsize=(12, 10))
    plt.title("Importance of coefficients")
    st.pyplot(fig)

with lr3:
    'error running it in streamlit not jupyter'
    '''
    lr_pred = lr.predict(X_test)
    sns.set_style("darkgrid")
    fig, ax = plt.subplot()
    fig = sns.regplot(lr_pred, y_test)
    st.pyplot(fig)
    '''


################################################################################

st.subheader("Part II: Prediction after dummifying categorical predictors")
st.write('before dummifying')
df_dum = df.drop(columns =['name'])
st.table(df_dum.head())

df_dum = pd.get_dummies(df_dum,
                        prefix=['neighbourhood', 'neighbourhood_group'],
                        drop_first=True)

'''
Dummified data sample
'''
st.table(df_dum.head())

################################################################################
corr_matrix = df_dum.corr()



d_top = corr_matrix['price'].sort_values(ascending=False)
st.write('Top 15 correlated variables to price are: \n', d_top.head(15))

#with lin2:
st.subheader("Linear Regression")

lin1, lin2, lin3 = st.columns((3))

with lin1:
    # Define
    X = df_dum.copy()
    y = X.pop('price')

    # diving data into trainning and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=10)

    # standardlize the train/test sets of data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                           columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test),
                          columns=X_train.columns)

    # Linear Regression
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_cv = cross_val_score(lr, X_train, y_train, cv=5)
    st.write('Cross-validated scores:', lr_cv)
    st.write('Average score:', lr_cv.mean())
    st.write('Trainning Score:', lr.score(X_train, y_train))
    st.write('Test Score:', lr.score(X_test, y_test))

with lin2:

    # collect the model coefficients in a dataframe
    df_coef = pd.DataFrame(lr.coef_, index=X_train.columns,
                           columns=['coefficients'])
    # calculate the absolute values of the coefficients
    df_coef['coef_abs'] = df_coef.coefficients.abs()



    coefs = pd.concat([df_coef['coefficients'].sort_values().head(10),
                         df_coef['coefficients'].sort_values().tail(10)])

    fig, ax = plt.subplots()
    coefs.plot(kind = "barh", figsize=(12, 10))
    plt.title("Importance of coefficients")
    st.pyplot(fig)

with lin3:
    'error running it in streamlit not jupyter'
    '''
    lr_pred = lr.predict(X_test)
    sns.set_style("darkgrid")
    plt.figure(figsize=(12,10))
    sns.regplot(lr_pred,y_test)
    '''

'**observation**'
'''
As you can observe from the cross validation scores, Linear Regression is not working using dummified data. And, it makes a lot of sense since
after the dummifications, most of the values in data turns into 0 which makes it a **sparse matrix**

As you can see from the correlatiion top 15 above, those are the variables that we should consider more when predicting the price!
So, I will regularization techniques to make use of the dummified data. 
'''
################################################################################
# Ridge
st.subheader('Regularizing the model')
reg1, reg2, reg3 = st.columns((3))

with reg1:
    "Ridge Regularization"
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV

    model = RidgeCV(alphas=np.logspace(-4, 4, 10), cv=5)
    model.fit(X_train, y_train)

    alpha = model.alpha_
    model = Ridge(alpha=alpha)

    # get cross validated scores
    scores = cross_val_score(model, X_train, y_train, cv=5)
    st.write("Cross-validated training scores:", scores)
    st.write("Mean cross-validated training score:", scores.mean())

    model.fit(X_train, y_train)
    st.write("Training Score:", model.score(X_train, y_train))
    st.write("Test Score:", model.score(X_test, y_test))


with reg2:
    # Lasso
    "Lasso Regularization"
    model = LassoCV(alphas=np.logspace(-4, 4, 10), cv=5)
    model.fit(X_train, y_train)

    alpha = model.alpha_
    model = Lasso(alpha=alpha)

    # get cross validated scores
    scores = cross_val_score(model, X_train, y_train, cv=5)
    st.write("Cross-validated training scores:", scores)
    st.write("Mean cross-validated training score:", scores.mean())

    model.fit(X_train, y_train)
    st.write("Training Score:", model.score(X_train, y_train))
    st.write("Test Score:", model.score(X_test, y_test))




with reg3:
    # Elastic net
    "Elastic net Regularization"
    model = ElasticNetCV(alphas=np.logspace(-4, 4, 10),
                         l1_ratio=np.array([.1, .5, .7, .9, .95, .99, 1]),
                         cv=5)
    # fit the model
    model.fit(X_train, y_train)

    alpha = model.alpha_
    model = ElasticNet(alpha=alpha)

    # get cross validated scores
    scores = cross_val_score(model, X_train, y_train, cv=5)
    st.write("Cross-validated training scores:", scores)
    st.write("Mean cross-validated training score:", scores.mean())

    model.fit(X_train, y_train)
    st.write("Training Score:", model.score(X_train, y_train))
    st.write("Test Score:", model.score(X_test, y_test))

'''
- Lasso gave the best R2 score so far. 
- And seeing the both training and testing scores, all the regurilized models seem not overfitting
'''


# NN
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (265,)),
        keras.layers.Flatten(),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(1,activation="linear")
    ]
)

model.compile(
    loss='mean_absolute_error',
    optimizer='adam',
    metrics=['mean_absolute_error']
)

history = model.fit(X_train,y_train,epochs=1000, validation_split=0.2, verbose=False)
st.write(model.evaluate(X_test, y_test))

nn1, nn2 = st.columns((2))

with nn1:

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper right')
    st.pyplot(fig)

with nn2:
    '''
    - Comparing the loss between train and test, even though I can further try to improve my model to decrease the loss,
    but so far model is not overfitting and running okay. 
    - As the number of epochs increase, the loss of train sets are descreasing but the validation set. 
    - I would try to find better nn models after the final week
    '''


















