#!/usr/bin/env python
# coding: utf-8

# # Problème: Nous avons été demandé de construire un systeme intelligent qui permet de generer un text
# ### Proposition de solution: Nous allons donc construire un modèle apprentissage profond(deep learning) pour résoudre le problème. Nous allons utiliser un LSTM Layer (couche ) , Embedding Layer qui permet de regrouper les mots selon leurs similitudes semantiques et Dense Layer qui permet de connecter les differents neuronnes entre elles.
# ### LSTM est une couche qui permet non seulement de faire des predictions en se basant sur les variables independantes mais aussi peut memoriser les informations antérieures. Comme exemple on va considerer le cas des series temporelles.
# ### Bidirectional Layer permet de faire des predictions en fonction des informations anterieures et posterieures

# ### Dans un premier temps nous allons importer les modules requis

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences


#
# ### Eclaircissement: Le Burkina Faso est un pays situé en Afrique de L'ouest avec 274.000 km2.Sa capital est ....... Comment notre système(modèle) va pouvoir prédire OUAGADOUGOU
# ### Afrique de l'ouest seulement ne permet pas de predire OUAGADOUGOU sinon qu'il y'a Mali aussi en Afrique de l'ouest😁😁😁. Alors le système va lire tout le text et comprendre le sens de la phrase afin de prendre une decision. En clair il a besoin des données pour apprendre quel mot vient après l'autre.
#

# In[2]:


with open("datasets/corpus.txt", encoding="utf-8") as f:
    data = f.read()


# In[3]:


corpus = data.lower().split("\n")  # Diviser le text en differentes phrase


# In[4]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(
    corpus
)  # Transformer le text en mot et son index: Example {1:"Crazy",2:"Love"}
tota_words = len(tokenizer.word_index) + 1


# In[5]:


print(f"Total words {tota_words}")


# In[6]:


input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[
        0
    ]  # Transformer le les phrases en sequences: Example: I Love you= [1,10,8]
    for i in range(1, len(token_list)):  # car I = 1, love = 10, you= 8
        n_gram_sequence = token_list[: i + 1]  # Deviser le text en ngrams
        input_sequences.append(n_gram_sequence)


# In[8]:


max_sequence_len = max(
    [len(x) for x in input_sequences]
)  # La taille de la sequence la plus longue


# In[10]:


max_sequence_len


# In[9]:


input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)


# In[7]:


print(input_sequences[:10])


# In[10]:


input_sequences[:5]


# In[11]:


max_sequence_len


# In[12]:


x = input_sequences[:, :-1]  # Variables dependantes
y = input_sequences[:, -1]  # Variable independantes


# In[13]:


y = tf.keras.utils.to_categorical(
    y, num_classes=tota_words
)  # Transformer notre variable independante en categorielle


# ### Construction de l'architucture du modèle

# In[26]:


# model = Sequential()
# model.add(Embedding(tota_words,240,input_length=max_sequence_len-1))
# model.add(Bidirectional(LSTM(150)))
# model.add(Dense(1012,activation='relu'))
# model.add(Dense(tota_words,activation="softmax"))
# model.compile( optimizer="adam",
#               loss="categorical_crossentropy",
#               metrics = ["accuracy"]
#              )

# history = model.fit(x,y,epochs=30,verbose=1)


# ### Allons y sauvegarder notre modèle pour utilisation postérieure ou déploiement

# In[15]:


# model2 = Sequential()
# model2.add(Embedding(tota_words,240,input_length=max_sequence_len-1))
# model2.add(Bidirectional(LSTM(150)))
# model2.add(Dense(1024,activation='relu'))
# model2.add(Dense(tota_words,activation="softmax"))
# model2.compile( optimizer="adam",
#               loss="categorical_crossentropy",
#               metrics = ["accuracy"]
#              )

# history = model2.fit(x,y,epochs=10,verbose=1)

# model2.save("text_generation_model2.h5")


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


# plt.figure(figsize=(10, 8))
# plt.plot(history.history["accuracy"], linewidth=8, color="b")
# plt.plot(history.history["loss"], linewidth=8, color="g")
# plt.xlabel("Iteration", fontsize=18)
# plt.ylabel("Accuracy & Loss", fontsize=18)
# plt.legend(["Accuracy", "Loss"], fontsize=18)
# plt.title("Accuracy & Loss Graph", fontsize=24)
# plt.show()


# # In[31]:


# plt.figure(figsize=(10, 8))
# plt.plot(history.history["accuracy"], linewidth=8, color="darkorange", label="Accuracy")
# plt.xlabel("Iteration", fontsize=18)
# plt.ylabel("Accuracy", fontsize=18)
# plt.legend(["Accuracy"], fontsize=18, loc="best")
# plt.title("Accuracy plot ", fontsize=24)
# plt.show()


# ### Après l'inspection du graphe ci-dessus nous pouvons conclure que le modèle fait bien en fonction des données qu'il a reçu (deux poèmes). Il peut predire chaque mot avec une certitude de 80.65%. Je peut aller prendre une bièrre🍾🍾🍾 pour ça😂😂😂😂

# ### La fonction suivante permet la génération du text. Pour ce faire,elle prend 2 parametres, seed_text un text racine ou initiale, c'est en fonction de ce text que le model va générer les phrases suivantes et le paramètre next_word qui est le nombre de caractères à générer

# In[11]:


mymodel = load_model("model/mymodel_text_generator.pkl")
mymodel2 = load_model("model/text_generator_.h5")


# In[12]:


import time


# In[13]:


def generate_text(seed_text, next_word):
    for _ in range(next_word):
        token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )
        predicted = np.argmax(mymodel2.predict(token_list, verbose=0))
        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word
    for i in seed_text.split(" "):
        for j in i:
            print(j, end="")
            time.sleep(0.2)
        print("", end=" ")
    return seed_text


# In[15]:


text = generate_text("""You and me will""", 80)


# In[35]:


model.save("model/mymodel_text_generator.pkl")


# ### Note:  Il faut noter que les phrases que notre système a construit n'a vraiment pas trop de sens parce que les données qui lui a été fourni ne sont vraiment pas enormes. Cependant une base de données assez large peut prendre des heures, des jours voir des mois d'entrainement du modèle en fonction de la capacité du GPU
