# -*- coding: utf-8 -*-
# Laget av Emma Dae, Morten Storvik og Mark Tzvetoslavov

import urllib.request
import pandas, re, random
import numpy as np
import sklearn.linear_model, sklearn.metrics, sklearn.model_selection

from sklearn.metrics import accuracy_score

# Vi valgte å implementere følgende bibliotekt
from collections import Counter
from sklearn import metrics

ORDFILER = {"norsk":"https://github.com/open-dict-data/ipa-dict/blob/master/data/nb.txt?raw=true",
        "arabisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ar.txt?raw=true",
        "finsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fi.txt?raw=true",
        "patwa":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/jam.txt?raw=true",
        "farsi":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fa.txt?raw=true",
        "tysk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/de.txt?raw=true",
        "engelsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_UK.txt?raw=true",
        "rumensk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ro.txt?raw=true",
        "khmer":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/km.txt?raw=true",
        "fransk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fr_FR.txt?raw=true",
        "japansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ja.txt?raw=true",
        "spansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/es_ES.txt?raw=true",
         "svensk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/sv.txt?raw?true",
         "koreansk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ko.txt?raw?true",
         "swahilisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/sw.txt?raw?true",
         "vietnamesisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/vi_C.txt?raw?true",
        "mandarin":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/zh_hans.txt?raw?true",
        "malayisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ma.txt?raw?true",
        "kantonesisk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/yue.txt?raw?true",
         "islandsk":"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/is.txt?raw=true"}

spraak = {"norsk": 0, "arabisk": 1, "finsk": 2, "patwa": 3, "farsi": 4, "tysk": 5, "engelsk": 6,
          "rumensk": 7, "khmer": 8, "fransk":9, "japansk": 10, "spansk": 11, "svensk": 12,
          "koreansk": 13, "swahilisk": 14, "vietnamesisk": 15, "mandarin": 16,
          "malayisk": 17, "kantonesisk": 18, "islandsk":19}

class LanguageIdentifier:
    """Logistisk regresjonsmodell som tar IPA transkripsjoner av ord som input,
    og predikerer hvilke språkene disse ordene hører til."""

    def __init__(self):
        """Initialiser modellen"""
        # selve regresjonsmodellen (som brukes all CPU-er på maskinen for trening)
        self.model = sklearn.linear_model.LogisticRegression(solver="liblinear", multi_class='ovr')
        self.symboler = []

    # (3)
    def train(self, transcriptions, languages):
        """Gitt en rekke med IPA transkripsjoner og en rekke med språknavn, tren
        den logistisk regresjonsmodellen. De to rekkene må ha samme lendgen"""
        X = self._extract_feats(transcriptions)
        y = self.sprak(languages)
        self.model.fit(np.array(X), np.array(y))

    # (4) Når vi er ferdig med å trene modellen vår, kan vi anvende den på nye fonetiske transkripsjoner.
    def predict(self, transcriptions):
        """Gitt en rekke med IPA transkripsjoner, finn ut det mest sansynnlige språket"""
        y = self._test(transcriptions)
        x = self.model.predict(y)
        res = self.konverter(x)
        return res

    def _test(self, transcriptions):

        matrise = np.zeros((len(transcriptions), len(self.symboler)), int)
        for i in range(len(transcriptions)):
            for j in range(len(self.symboler)):
                if self.symboler[j] in transcriptions[i]:
                    matrise[i][j] = 1
        return matrise

    def konverter(self, languages):
        liste = []
        for elementer in languages:
            for x, y in spraak.items():
                if y == elementer:
                    liste.append(x)
        return liste



    # (1) Det første skrittet er å lage en liste over alle IPA-symboler som finnes i treningssettet
    def _extract_unique_symbols(self, transcriptions, min_nb_occurrences=10):
        """Gitt en rekke med IPA fonetiske transkripsjoner, ektraher en liste med alle IPA
        symboler som finnes i transkripsjonene og forekommer minst min_nb_occurrences."""

        liste = {}
        for element in transcriptions:
            for symbol in Counter(element):
                if symbol not in liste:
                    liste[symbol] = 1
                else:
                    sym = liste[symbol]
                    sym += 1
                    liste[symbol] = sym

        resultat = []
        for symbol, forekomster in liste.items():
            if forekomster >= min_nb_occurrences:
                resultat.append(symbol)
        return resultat

    # (2)
    def _extract_feats(self, transcriptions):
        """Gitt en rekke med IPA transkripsjoner, ekstraher en matrise av størrelse |T|x|F|,
        hvor |T| er antall transkripsjoner, og |F| er antall features brukt i modellen."""

        # m -> unike fonetiske symboler
        # n -> fonetiske transkripsjoner

        symboler = self._extract_unique_symbols(transcriptions)
        self.symboler = symboler
        matrise = np.zeros((len(transcriptions), len(symboler)), int)

        for i in range(len(transcriptions)):
            for j in range(len(symboler)):
                if symboler[j] in transcriptions[i]:
                    matrise[i][j] = 1

        return matrise
        #raise NotImplementedException()

    # En metode som vi valgte å implementere for å hente frem spraak posisjonen.
    def sprak(self, languages):
        liste = []
        for elementer in languages:
            x = spraak[elementer]
            liste.append(x)
        return np.array(liste)

    def analyse(self):
        return


    def evaluate(self, transcriptions, languages):
        """Gitt en rekke med IPA transkripsjoner og en rekke med språknavn, evaluer hvor godt
        modellen fungerer ved å beregne:
        1) accuracy
        2) precision, recall og F1 for hvert språk
        3) micro- og macro-averaged F1.
        """

        transkripsjoner = self._test(transcriptions)
        evaluation = self.model.predict(transkripsjoner)
        res = self.sprak(self.konverter(evaluation))
        y = self.sprak(languages)
        accuracy = accuracy_score(res, y)
        print("- Accuracy: " + str(accuracy) + " -\n\n")
        # See API fra sklearn.metrics for å finne ut hvordan dette kan gjøres!
        # raise NotImplementedException()
        classification = metrics.classification_report(res, y, digits=3, output_dict = True)

        print("- precision, recall og F1 for hvert språk - \n")
        print(" " * 16+ " precision" + " " * 4 + "recall" + " " * 5 + "F1-(micro)" + " " * 2)
        for language, metric in classification.items():
            sentence = ""
            if (language.isdigit()):
                for lan, num in spraak.items():
                    if num == int (language):
                        sentence = lan
                        while (len(sentence) < 12):
                            sentence += " "
                        for elementer, verdi in metric.items():
                            if elementer != "support":
                                sentence += " " * 7 + '%.3f' % verdi
                print(sentence)
            else:
                if language == "macro avg":
                    print("\n- macro (f1-score) -")
                    print("f1-score: " + str(metric['f1-score']))




def extract_wordlist(max_nb_words_per_language=20000):
    """
    Laster ned fra Github ordlister med ord og deres phonetiske transkripsjoner i flere språk.
    Ordlistene er deretter satt sammen i en pandas DataFrame, og delt i en treningsett og en testsett.
    """

    full_wordlist = []
    for lang, wordfile in ORDFILER.items():

        print("Nedlasting av ordisten for", lang, end="... ")
        data = urllib.request.urlopen(wordfile)

        wordlist_for_language = []
        for linje in data:
            linje = linje.decode("utf8").rstrip("\n")
            word, transcription = linje.split("\t")

            # Noen transkripsjoner har feil tegn for "primary stress"
            transcription = transcription.replace("\'", "ˈ")

            # vi tar den første transkripsjon (hvis det finnes flere)
            # og fjerner slashtegnene ved start og slutten
            match = re.match("/(.+?)/", transcription)
            if not match:
                continue
            transcription = match.group(1)
            wordlist_for_language.append({"ord":word, "IPA":transcription, "språk":lang})
        data.close()

        # Vi blander sammen ordene, og reduserer mengder hvis listen er for lang
        random.shuffle(wordlist_for_language)
        wordlist_for_language = wordlist_for_language[:max_nb_words_per_language]

        full_wordlist += wordlist_for_language
        print("ferdig!")

    # Nå bygger vi en DataFrame med alle ordene
    full_wordlist = pandas.DataFrame.from_records(full_wordlist)

    # Og vi blander sammen ordene i tilfeldig rekkefølge
    full_wordlist = full_wordlist.sample(frac=1)

    # Lage et treningssett og en testsett (med 10% av data)
    wordlist_train, wordlist_test = sklearn.model_selection.train_test_split(full_wordlist, test_size=0.1)
    print("Treningsett: %i eksempler, testsett: %i eksempler"%(len(wordlist_train), len(wordlist_test)))

    return wordlist_train, wordlist_test




#######################
# Brukseksempel:
#######################
if __name__ == "__main__":

    # Vi laster ned dataene (vi trenger kun å gjøre det én gang)
    train_data, test_data = extract_wordlist()

    # Vi teller antall ord per språk
    print("Statistikk over språkene i treningsett:")
    print(train_data.språk.value_counts())
    print("Første 30 ord:")
    print(train_data[:30])

    # Vi bygge og trene modellen
    model = LanguageIdentifier()
    transcriptions = train_data.IPA.values
    languages = train_data.språk.values
    model.train(transcriptions, languages)

    # Vi kan nå test modellen på nye data
    predicted_langs = model.predict(["konstituˈθjon", "ɡrʉnlɔʋ", "stjourtnar̥skrauːɪn", "bʊndɛsvɛɾfaszʊŋ"])
    print("Mest sansynnlige språk for ordene:", predicted_langs)

    # Til slutt kan vi evaluere hvor godt modellen fungerer på testsett
    model.evaluate(test_data.IPA.values, test_data.språk.values)
