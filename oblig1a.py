# IN2110 Oblig 1 pre-kode
# Klasser og funksjoner fra scikit-learn som vi skal bruke i obligen
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Norec dataset

from in2110.corpora import norec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Hjelpefunksjoner for visualisering
from in2110.oblig1 import scatter_plot

def prepare_data(documents):
    """Tar inn en iterator (kan være en liste) over dokumenter fra norec
    og returnerer to lister:
    - data   : En liste over dokument-tekstene.
    - labels : En liste over hvilken kategori dokumentet tilhører.
    Begge listene skal være like lange og for dokumentet i data[i]
    skal vi kunne finne kategorien i labels[i].
    """

    # Din kode her
    # Din kode her
    data = []
    labels = []

    for text in documents:
        category = text.metadata['category']
        if (category == 'games' or category == 'restaurants' or category == 'literature'):
            data.append(text)
            labels.append(category)

    # test for å se om metoden er implementert riktig
    # print("Data: " + str(len(data))) # Data: 4662
    # print("Labels: " + str(len(labels))) # Labels: 4662

    return data, labels

def tokenize(text):
    """Tar inn en streng med tekst og returnerer en liste med tokens."""

    # Å splitte på mellomrom er fattigmanns tokenisering. Endre til noe
    # bedre!


    # Oppgave a
    # Tokeniser trenings-settet med den originale funksjonen og rapporter
    # hvor mange tokens og hvor mange ordtyper du får.
    # return text.split()

    # Oppgave b

    # Endre funksjonen til å bruke en bedre tokenizer, f.eks word_tokenize
    # i NLTK. Rapporter antall tokens og ordtyper for denne også.
    # return word_tokenize(text)

    # Oppgave c
    # Prøv andre typer pre-prosessering, som f.eks. å gjøre om alle ord til
    # små bokstaver ...
    sentences = word_tokenize(text.lower())
    return sentences

    # Oppgave c (forts.)
    """
    stop_words = stopwords.words('norwegian')

    ord_text = []
    lemmatizer = WordNetLemmatizer()


    tekst = re.sub(r'[^\w\s]', '', text)
    words = nltk.word_tokenize(tekst.lower())
    for ord in words:
       if ord not in stop_words:
          ord_text.append(lemmatizer.lemmatize(ord))
    return ord_text
    """

class Vectorizer:
    def __init__(self):
        """Konstruktør som tar inn antall klasser som argument."""

        self.vectorizer = CountVectorizer(lowercase = False, tokenizer = tokenize, max_features = 5000)
        self.tfidf = TfidfTransformer()

    def vec_train(self, data):
        #vector_fidf = vec_tfidf.format(-1, 1)
        """Tilpass vektorisereren til treningsdata. Returner de vektoriserte
        treningsdataene med og uten tfidf-vekting.
        """
        # Din kode her
        # Tips: Bruk fit_transform() for å spare kjøretid.

        vec = self.vectorizer.fit_transform(text for text, meta in data)
        vec_tfidf = self.tfidf.fit_transform(vec)

        return vec, vec_tfidf

    def vec_test(self, data):
        """Vektoriser dokumentene i nye data. Returner vektorer med og uten
        tfidf-vekting.
        """
        vec = self.vectorizer.transform(text for text, meta in data)
        vec_tfidf = self.tfidf.transform(vec)


        return vec, vec_tfidf

def create_knn_classifier(vec, labels, k):
    """Lag en k-NN-klassifikator, tren den med vec og labels, og returner
    den.
    """

    clf = KNeighborsClassifier(k)
    clf.fit(vec, labels)

    return clf


# Treningsdata
train_data, train_labels = prepare_data(norec.train_set())

# Valideringsdata
dev_data, dev_labels = prepare_data(norec.dev_set())

# Testdata
test_data, test_labels = prepare_data(norec.test_set())

# Din kode her

# +--------------------------------------+
# + OPPGAVE 1 - DATA OG PRE-PROSESSERING +
# +--------------------------------------+

# b) Pre-prosessering
#   - Dersom vi tokeniserer treningsettet med den orignale funksjonen,
#     får vi totalt 1829550 tokens og 186519 ordtyper.
#
#   - Dersom vi endrer funksjonen til å bruke en bedre tokenizer, f.eks
#     word_tokenize i NLTK får vi 2086943 tokens og 127814 ordtyper
#
#   - Dersom vi prøver med word_tokenize i tillegg til at vi gjør om alle ord
#     til små bokstaver får vi 2085877 tokens og 115967 ordtyper
#
#   - Til slutt, hvis vi anvender ulike metoder, som for eksempel
#     unødvendig tegn, fjerne stoppeord, og bruker lemmatisering, får
#     vi 998071 tokens og 112445.
#
#   Hvilken tokenisering gir lavest antall ordtyper?
#   Etter å ha analysert svarene på de ulike tokeniseringer, har vi kommet fram
#   til at den siste tokenisering, det vil si, den tokeniseringen som tar i bruke
#   word_tokenize.lower(), stopwords, regex_uttrykk, osv gir den lavest antall
#   ordtyper. Dette er noe som var forvent, siden vi har tatt i bruk ulike
#   metoder for å bli kvitt med ord som ikke inneholder noe leksikalsk innhold,
#   eller ord/tegn som ikke kommer til å brukes videre.
#   (OBS: Vi har bruk den nest sist tokeniserer (word_tokenize.lower() for å
#         løse de andre oppgaver.))

print("\n+--------------------------------------+")
print("+ OPPGAVE 1 - DATA OG PRE-PROSESSERING +")
print("+--------------------------------------+\n")
print("b) Pre-prosessering")

tokens = []
for dokumenter in train_data:
   for words in tokenize(dokumenter.text):
       tokens.append(words)

print("Antall tokens  : " + str(len(tokens)))
print("Antall ordtyper: " + str(len(set(tokens))) + "\n")

# c) Statistikk
#    Beregn antall dokumenter per kategori ('games', 'restaurants' og
#    'literature') i treningsettet.

games, restaurants, literature = 0, 0, 0
for i in range(len(train_labels)):
    if (train_labels[i] == 'games'):
        games += 1
    elif (train_labels[i] == 'restaurants'):
        restaurants += 1
    else:
        literature += 1

# Fordelingen mellom de ulike kategoriene er litt ujevnt, siden i restaurants
# kategorien får vi bare 428 dokumenter utifra de 4662 filene. Dette tilsvarer
# 9.18 % av alle filer i treningssettet. Som en konsekvens, vil vi ikke
# klare helt å kategorisere riktig klasse dersom vi setter en fil som er ment
# til å tilhøre restaurants-"klassen".
# literature-kategorien inneholder 60.5% av alle treningsfilene som vi har
# oppgitt og games-kategorien inneholder 30.3%. Dette vil, uten tvil, være et
# problem for oss siden i prinsippet må vi ha like mange filer for hver kategori
# for å være i stand til å få et mer tilnærmet og riktig resultat dersom vi
# setter inn filer som vi ikke har sett før (f.eks test-data).

print("\nc) Statistikk")
# Diskuter kort fordelingen mellom kategorine
print(" - Antall dokumenter - ")
print("games      : " + str(games)) # 1413
print("restaurants: " + str(restaurants)) # 428
print("literature : " + str(literature)) # 2821
print("Summen     : " + str(games + restaurants + literature) + "\n") # 4662
print(" - Prosent-representasjon - ")
print("Games      : " + '{0:.2f}'.format(games / (games + restaurants + literature)))
print("Restaurants: " + '{0:.2f}'.format(restaurants / (games + restaurants + literature)))
print("literature : " + '{0:.2f}'.format(literature / (games + restaurants + literature)))


# +--------------------------------------+
# +  OPPGAVE 2 - DOKUMENTREPRESENTASJON  +
# +--------------------------------------+

print("\n+--------------------------------------+")
print("+  OPPGAVE 2 - DOKUMENTREPRESENTASJON  +")
print("+--------------------------------------+\n")

# Oppgave 2 - Dokumentrepresentasjon
# Visualiser dokumentvektorene for trenings-settet. Lagre grafen som en fil
# og legg ved i rapporten. Beskriv og diskuter hva du ser.

vec1 = Vectorizer()
train_vector, tfidf_vector = vec1.vec_train(train_data)
scatter_plot(train_vector, train_labels)
scatter_plot(tfidf_vector, train_labels)

validering_vector, tfidf_validering_vector = vec1.vec_test(dev_data)
scatter_plot(validering_vector, dev_labels)
scatter_plot(tfidf_validering_vector, dev_labels)

# +--------------------------------------+
# +  OPPGAVE 3 - DOKUMENTREPRESENTASJON  +
# +--------------------------------------+

print("\n+--------------------------------------+")
print("+     OPPGAVE 3 - KLASSIFISERING       +")
print("+--------------------------------------+\n")

# Lag dokumentvektorer for dokumentene i valideringssettet. Deretter tren
# en klassifikator for forskjellige verdier av k med og uten tfidf, og
# beregn accuracy for hver av dem, ved å f.eks teste med k fra 1 til 20.

# Kommentar:
# Det vi gjør her er å trene en knn-klassifikator med og uten tf-idf vektorer
# og bruke valideringssettet for å se hvilken som gir best accuracy.

# --- UTEN TF-IDF VEKTING ---
print("b) Evaluering")
print("kNN-Klassifikator uten tf-idf vekting: ")
knn1 = create_knn_classifier(train_vector, train_labels, 1)
knn10 = create_knn_classifier(train_vector, train_labels, 10)
knn20 =  create_knn_classifier(train_vector, train_labels, 20)

predikert1 = knn1.predict(validering_vector)
predikert10= knn10.predict(validering_vector)
predikert20 = knn20.predict(validering_vector)

print("k = 1 ---> " + str(accuracy_score(dev_labels, predikert1)))
print("k = 10 ---> " + str(accuracy_score(dev_labels, predikert10)))
print("k = 20 ---> " + str(accuracy_score(dev_labels, predikert20)))

print("\nkNN-Klassifikator med tf-idf vekting: ")
knn1 = create_knn_classifier(tfidf_vector, train_labels, 1)
knn10 = create_knn_classifier(tfidf_vector, train_labels, 10)
knn20 =  create_knn_classifier(tfidf_vector, train_labels, 20)

predikert1 = knn1.predict(tfidf_validering_vector)
predikert10= knn10.predict(tfidf_validering_vector)
predikert20 = knn20.predict(tfidf_validering_vector)

print("k = 1 ---> " + str(accuracy_score(dev_labels, predikert1)))
print("k = 10 ---> " + str(accuracy_score(dev_labels, predikert10)))
print("k = 20 ---> " + str(accuracy_score(dev_labels, predikert20)))

# c) Testing
#    Lag dokumentvektorer for testsettet og tren en klassifikator med samme
#    valg for k-verdi og vekting (tfidf eller ikke) som den med høyest
#    accuracy i forrige deloppgave.



print("\nc) Testing")
print("kNN-Klassifikator uten tf-idf vekting: ")
test_vector, tfidf_test_vector = vec1.vec_test(test_data)
knn1 = create_knn_classifier(train_vector, train_labels, 1)
knn10 = create_knn_classifier(train_vector, train_labels, 10)
knn20 =  create_knn_classifier(train_vector, train_labels, 20)

predikert1 = knn1.predict(test_vector)
predikert10= knn10.predict(test_vector)
predikert20 = knn20.predict(test_vector)

print("k = 1 ---> " + str(accuracy_score(test_labels, predikert1)))
print("k = 10 ---> " + str(accuracy_score(test_labels, predikert10)))
print("k = 20 ---> " + str(accuracy_score(test_labels, predikert20)))


print("\nkNN-Klassifikator med tf-idf vekting: ")
test_vector, tfidf_test_vector = vec1.vec_test(test_data)
knn1 = create_knn_classifier(tfidf_vector, train_labels, 1)
knn10 = create_knn_classifier(tfidf_vector, train_labels, 10)
knn20 =  create_knn_classifier(tfidf_vector, train_labels, 20)

predikert1 = knn1.predict(tfidf_test_vector)
predikert10= knn10.predict(tfidf_test_vector)
predikert20 = knn20.predict(tfidf_test_vector)

print("k = 1 ---> " + str(accuracy_score(test_labels, predikert1)))
print("k = 10 ---> " + str(accuracy_score(test_labels, predikert10)))
print("k = 20 ---> " + str(accuracy_score(test_labels, predikert20)))
