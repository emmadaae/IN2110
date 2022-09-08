# Oppgave 3 - Evaluering
# Vi skal nå jobbe med evaluering av parseren og skal derfor parse
# development datasettet no_bokmaal-ud-dev.conllu med parseren vi
# har trent. Den ferdig trente modellen fra forrige oppgave kan vi laste
# inn i spaCy.

import spacy
from in2110.conllu import ConlluDoc
nb = spacy.load("modellmappe/model-best")


def unlabeled(file):

    # UD-filene er i CoNLL-U-formatet, og kan ikke leses direkte av spaCy.
    # For å lese inn har vi brukt klassen ConlluDoc som allerede er importert
    # i pre-koden. Derretter har vi konvertert disse til spaCy-objekter.

    conllu_dev = ConlluDoc.from_file(file)
    dev_docs = conllu_dev.to_spacy(nb)

    # Dette gir oss en liste med Doc-objekter som hver representer en setning.
    # Hvert enkelt Doc-objekt er igjen en liste med token-objekter (med relevante
    # attributter som head, i og dep_, som angir henholdvis hode, indeks og
    # dependsrelasjonen for et gitt token).

    # Disse objektene inneholder nå gull-dataene for devsettet vårt. For å evaluere
    # trenger vi også setninger uten annotasjoner som vi kan parse med parseren
    # vi har laget.

    tokens = [y for x in dev_docs for y in x]
    head = [x.head for x in tokens]
    final = set(head)

    dev_docs_unlabeled = conllu_dev.to_spacy(nb, keep_labels=False)
    doc = dev_docs_unlabeled[13]
    doc.is_parsed


    parsed = []
    for x in dev_docs_unlabeled:
        parsed.append(nb.parser(x))

    return dev_docs, dev_docs_unlabeled

# Her ligger attachment_score funksjonen vår som vi har utvidet for å så kunne
# beregne UAS (unlabeled attachment score) og LAS (labeled attachment score)
# i siste bitten av oppgaven.

def attachment_score(gull, liste):
    headT, headP, depT, depP = [], [], [], []

    for setning in gull:
        for x in setning:
            headT.append(x.head)
            depT.append(x.dep_)

    for prediksjon in liste:
        for y in prediksjon:
            headP.append(y.head)
            depP.append(y.dep_)

    ord, hodeR, depR = len(headT), 0, 0
    for y, x in enumerate(headT):
        if str(x) == str(headP[y]):
            hodeR += 1
            if str(depT[y]) == str(depP[y]):
                depR += 1

    return hodeR/ord, depR/ord


print("TEST DATA: no_bokmaal-ud-dev.conllu")
dev_docs_bokmaal, dev_docs_unlabeled_bokmaal = unlabeled("no_bokmaal-ud-dev.conllu")
no_bokmaal = attachment_score(dev_docs_bokmaal, dev_docs_unlabeled_bokmaal)
print(no_bokmaal)

print("TEST DATA: no_nynorsk-ud-dev.conllu")
dev_docs_nynorsk, dev_docs_unlabeled_nynorsk = unlabeled("no_nynorsk-ud-dev.conllu")
no_nynorsk = attachment_score(dev_docs_nynorsk, dev_docs_unlabeled_nynorsk)
print(no_nynorsk)

print("TEST DATA: no_nynorsklia-ud-dev.conllu")
dev_docs_nynorsklia, dev_docs_unlabeled_nynorsklia= unlabeled("no_nynorsklia-ud-dev.conllu")
no_nynorsklia = attachment_score(dev_docs_nynorsklia, dev_docs_unlabeled_nynorsklia)
print(no_nynorsklia)
