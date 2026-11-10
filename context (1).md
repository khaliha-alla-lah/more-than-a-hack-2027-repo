# More Than a Hack 2027 — Phase 1
## Handover context: possession inference pipeline

**État au : 11 novembre 2026**  
**Objectif immédiat :** reprendre le travail sur le pipeline de possession, améliorer les résultats de manière reproductible et livrer une soumission conforme avant le **17 novembre 2026 à 23:59 CET**.

> Ce document est une passation basée sur l’état et les résultats communiqués par l’équipe au 11 novembre 2026. Il distingue les résultats déjà observés des hypothèses et des expériences encore à faire. Le code local, les manifests de split et les fichiers de configuration restent la source de vérité : vérifiez-les avant toute modification.

---

## 1. Le challenge en bref

Le challenge demande de prédire un label pour chacune des **199 319 frames** de la vidéo tactique du match FC Barcelona–Athletic Club (LaLiga 2024–2025).

| Label | Signification |
|---|---|
| `H` | Barcelone contrôle le ballon pendant le jeu actif |
| `A` | Athletic Club contrôle le ballon pendant le jeu actif |
| `D` | Jeu arrêté / ballon inactif |
| `X` | Abstention (à ne pas considérer comme une stratégie d’abstention avantageuse sans vérifier le scorer officiel) |

La métrique la plus importante est **`spellF1`** : elle évalue la reconstruction des possessions comme segments temporels, pas uniquement la classification frame par frame. `macroF1` est également rapportée et intervient dans le classement/départage.

La baseline officielle publiée pour le match complet est de **0,2433 spellF1** (`smooth50`). Ce score n’est pas directement comparable à un score obtenu sur un autre dataset ou un autre match.

### Contraintes importantes

- Ne jamais annoter manuellement la vidéo Barcelona–Athletic pour créer ou corriger des labels.
- Ne pas utiliser le ground truth privé du match ni des données humaines spécifiques à ce match.
- Ne pas coder de corrections propres à des frames identifiées manuellement dans le match cible.
- Les données et vidéos du match doivent rester locales et ne doivent pas être publiées dans le dépôt.
- Les cinq premières minutes de labels publics servent aux contrôles autorisés prévus par les règles ; ne pas les utiliser pour entraîner, sélectionner des hyperparamètres ou optimiser le modèle.
- Les données externes annotées d’autres matchs peuvent servir à l’entraînement et à la validation, sous réserve de licence et de conformité aux règles.

Consulter impérativement `RULES.md`, `EVALUATION.md`, `DATA.md`, `CHALLENGE.md` et `SUBMISSION.md` dans le dépôt officiel avant la soumission.

---

## 2. Données utilisées

### Données du match cible

Le package organisateur contient notamment :

- **Tactical video** : vidéo de référence, 199 319 frames à 25 FPS.
- **Tactical ATD** : tracking automatique des joueurs et du ballon, avec des positions normalisées et des valeurs manquantes.
- **Tactical homography** : correspondance image/terrain. Elle sert surtout à projeter des détections visuelles sur le terrain ; les coordonnées ATD sont déjà dans l’espace terrain.
- **Smart Tags** : annotations tactiques automatiques, utiles comme contexte/debug, mais bruitées et pas équivalentes à une vérité terrain de possession frame par frame.
- **Broadcast video / ATD / homography** : source secondaire, avec changements de caméra, zooms, replays et problème d’alignement avec la vidéo tactique.

Constats communiqués dans l’analyse du dataset :

- Le ballon est absent du tracking tactique sur environ **41 % des frames évaluées**.
- Le nombre de joueurs trackés est variable et les identifiants de tracking ne sont pas des identités fiables et permanentes.
- Les gaps de ballon peuvent être longs ; ils ne signifient pas automatiquement que le jeu est arrêté.
- Les Smart Tags sont des indices contextuels, pas des labels exacts de `D` ou de l’équipe en possession.

### Données externes : Sportec Open DFL / IDSSE

L’équipe utilise les données IDSSE / Sportec Open DFL, qui comprennent des matchs professionnels avec tracking et annotations temporelles. Les champs clés identifiés sont :

- `BallStatus` : actif / inactif.
- `BallPossession` : équipe à domicile / équipe à l’extérieur.

Ces champs constituent une supervision externe utile, mais il faut vérifier leur sémantique exacte et leur compatibilité avec la définition du challenge. En particulier, **une valeur de possession manquante/inconnue ne doit pas être convertie automatiquement en arrêt de jeu**.

Le DFL est conservé provisoirement : son utilité est plausible et ses labels sont précieux, mais le transfert DFL → tracking Metrica n’est pas encore démontré de façon robuste.

---

## 3. Pipeline candidat actuel

L’architecture générale à conserver comme référence est : **deux classifieurs LightGBM + décodage temporel Viterbi/HMM**.

### Classifieur 1 — Live / Dead

- La variante actuellement retenue utilise principalement le **mouvement collectif et la forme spatiale des joueurs**.
- Le modèle avec les signaux de ballon pour `Live/Dead` a obtenu de meilleurs résultats sur certains tests DFL, mais son comportement sur le tracking cible a été jugé peu plausible ; il n’a donc pas été retenu dans la variante déployée.
- Le statut d’activité doit tenir compte du fait que des joueurs peuvent bouger pendant un arrêt et rester relativement immobiles pendant une phase active.

### Classifieur 2 — Home / Away control

- Utilise davantage d’information de tracking : position du ballon quand elle est disponible, distances joueur-ballon, mouvement relatif, proximité des joueurs et observations avant/après les gaps.
- Les features doivent être robustes aux identifiants de tracking fragmentés ; privilégier les caractéristiques collectives et géométriques plutôt que l’identité individuelle.

### Fusion et décodage

Les probabilités des deux classifieurs sont combinées pour obtenir des scores pour `H`, `A` et `D`, puis une séquence est décodée par Viterbi séparément pour chaque mi-temps.

Le décodeur :

- autorise les turnovers directs `H ↔ A` ;
- autorise les séquences `H → D → H` et `A → D → A` ;
- pénalise les changements de possession insuffisamment étayés ;
- ne doit pas imposer une durée minimale arbitraire ni fusionner automatiquement les possessions courtes.

L’intervalle pré-kickoff de seconde mi-temps explicitement fourni par les métadonnées organisateur est traité comme dead-ball. Ce n’est pas une correction manuelle de frames observées.

### Vidéo

- La soumission actuelle utilise **le tactical ATD uniquement**.
- Les Smart Tags sont actuellement utilisés comme aide locale au debugging, pas comme vérité terrain.
- Une première vérification du broadcast n’a pas permis de distinguer de façon fiable les phases de jeu en direct des replays ; le broadcast est donc écarté pour le moment.
- La récupération automatique du ballon dans la vidéo tactique reste une expérience possible, uniquement si les erreurs du tracking demeurent le facteur limitant après correction des features et de l’augmentation.

---

## 4. Résultats déjà rapportés — ne pas mélanger les protocoles

Les chiffres ci-dessous proviennent des rapports et expériences internes communiqués par l’équipe. Ils n’ont pas été reproduits indépendamment dans ce document.

### 4.1 Deux matchs de validation externe, tracking dégradé

| Match | Nearest-player baseline | Pipeline candidat |
|---|---:|---:|
| `J03WN1` | 0,1971 | **0,2853** |
| `J03WQQ` | 0,2313 | **0,3949** |
| Moyenne | 0,2142 | **0,3401** |

Le gain moyen annoncé est de **+0,1259 spellF1** par rapport à nearest-player sur ces deux matchs et ce protocole de dégradation.

**Limite :** ces deux matchs ont déjà été utilisés pour éclairer les discussions et décisions. Ils ne doivent plus être présentés comme un test totalement intact si de nouveaux réglages sont faits en fonction de leurs erreurs.

### 4.2 Scénario de stress

| Mesure moyenne rapportée | Nearest-player | Pipeline candidat |
|---|---:|---:|
| `spellF1` | 0,1984 | 0,2201 |

Le gain sous stress est faible (+0,0217). Il n’établit pas une robustesse générale ; il faut examiner le nombre de matchs, les scénarios de corruption et la variabilité par match.

### 4.3 Sweep de développement : architectures

Moyenne rapportée sur les scénarios de développement dégradé/stress :

| Variante | Moyenne `spellF1` rapportée |
|---|---:|
| Deux modèles binaires avec full features | **0,3160** |
| Modèle trois classes | 0,3007 |
| Variante collective-status retenue pour le transfert cible | 0,2692 |

Ces chiffres utilisent un protocole de développement différent des résultats de validation `J03WN1` / `J03WQQ`. **Ne pas les comparer directement** aux valeurs 0,3401 / 0,2201.

Le modèle full-feature avait le meilleur score de développement, mais le classifieur Live/Dead full-feature a été écarté de la variante déployée car son transfert vers les entrées du match cible paraissait peu plausible. Cette décision doit être réévaluée après correction de l’augmentation.

### 4.4 Possessions courtes : faiblesse majeure

Définition observée dans le rapport : possessions de **13 à 37 frames** (environ 0,5 à 1,5 s), évaluées au seuil temporel IoU 0,5.

| Étape | Match 1 | Match 2 | Total |
|---|---:|---:|---:|
| Détection avant décodage | 6/96 | 3/46 | **9/142 (6,34 %)** |
| Après Viterbi | 1/96 | 2/46 | **3/142 (2,11 %)** |

Interprétation prudente :

- Le problème existe **déjà avant le décodage** : le classifieur ne détecte que 9 épisodes sur 142.
- Viterbi en élimine ensuite une partie supplémentaire.
- Un échec de matching peut venir d’une possession manquée, d’une mauvaise équipe, d’une fragmentation ou d’une mauvaise frontière ; le seul taux de matching ne permet pas de distinguer ces causes.
- Ne pas supprimer Viterbi sans test : il réduit également des segments parasites et améliore le score global.

---

## 5. Découverte critique de l’audit : l’augmentation déforme les vitesses

L’audit du 11 novembre a identifié un confondant sérieux dans la corruption synthétique actuelle.

La dégradation ajoute un bruit positionnel indépendant à chaque frame. Les features calculent ensuite les vitesses à partir des positions successives à 25 FPS. Le bruit indépendant crée donc des déplacements artificiels. Lisser la vitesse après le calcul ne supprime pas le biais positif introduit par la norme du déplacement.

### Mesures communiquées — J03WMX, première mi-temps, vitesse collective domicile

| Entrée | Médiane inactive | Médiane active |
|---|---:|---:|
| Originale | 0,993 m/s | 2,277 m/s |
| Dégradation actuelle | 3,706 m/s | 4,213 m/s |
| Stress actuel | 6,001 m/s | 6,204 m/s |

L’écart entre les médianes active/inactive passe de 1,284 m/s (original) à 0,507 m/s (dégradé), puis 0,203 m/s (stress).

Un test sur des positions synthétiques stationnaires a produit des vitesses mesurées moyennes de 3,540 m/s et 5,927 m/s pour les deux niveaux de bruit, après le cutoff utilisé dans les features.

La médiane de vitesse collective domicile sur la première mi-temps cible est de 1,481 m/s, mais sans vérité terrain cible il est impossible de séparer les frames actives et inactives.

**Ce que cela démontre :** la corruption actuelle déforme une feature centrale pour la détection des arrêts et fait se rapprocher les statistiques de mouvement actives et inactives.

**Ce que cela ne démontre pas :** la part exacte de perte de `spellF1` causée par cette corruption. Il faut isoler les effets avec des ablations.

### Correction à tester (sans changer la production avant mesure)

1. Séparer les corruptions : aucune corruption, gaps ballon seuls, pertes de joueurs seules, bruit de coordonnées seul, puis combinaisons.
2. Comparer le jitter indépendant actuel à une erreur temporellement corrélée (par exemple AR(1)), calibrée sur des propriétés observées du tracking.
3. Estimer les vitesses après filtrage robuste des positions, ou comparer avec les vitesses natives du tracker après vérification de leurs unités/qualité.
4. Mesurer les distributions de vitesse (P10/P25/P50/P75/P90), pas seulement les médianes.
5. Reproduire les gaps continus et la perte de joueurs avec plusieurs graines fixes ; vérifier les gaps effectivement réalisés, car des masques qui se chevauchent changent leur distribution.
6. Tester une missingness liée à la géométrie/occlusion en plus des gaps uniformes, sans jamais utiliser les labels pour choisir où corrompre les entrées.

Ne pas choisir un nouveau niveau de bruit simplement parce qu’il améliore le score : il doit aussi être plausible par rapport aux propriétés du tracking cible.

---

## 6. Ce qui reste incertain

1. **Transfert DFL → Metrica :** pas encore démontré de manière indépendante et robuste.
2. **Sémantique des labels :** `BallStatus` / `BallPossession` peuvent être proches de la cible, mais il faut vérifier le traitement des ballons libres, déviations, touches, contrôles brefs et valeurs inconnues.
3. **Possessions courtes :** rappel extrêmement faible avant comme après décodage.
4. **Dégradation synthétique :** le jitter indépendant actuel déforme la vitesse et probablement la séparation Live/Dead ; son impact causal sur le score reste à mesurer.
5. **Robustesse :** les deux matchs de validation déjà examinés ne constituent plus un holdout totalement vierge.
6. **Forks concurrents :** les chiffres annoncés par Blaze2fire, ScoreAdvise et SAM Analytics n’ont pas été reproduits sur un benchmark commun. Les rapports d’autres équipes sont des pistes, pas des preuves.
7. **Vidéo :** aucun gain quantifié de la récupération du ballon n’a encore été établi.
8. **Classe `D` :** il manque encore des tableaux dédiés complets precision/recall/F1 et une analyse des retards de reprise.

---

## 7. Plan de reprise recommandé — dans cet ordre

### P0 — Geler l’état actuel et sécuriser la reproductibilité

- Relever le commit Git exact.
- Calculer les hashes du code, des poids, des dépendances, des manifests de splits et des datasets utilisés.
- Archiver les graines de corruption et les commandes exécutées.
- Garder le candidat actuel comme référence immuable.
- Vérifier que la soumission actuelle peut être régénérée exactement.

Ne pas écraser le candidat courant pendant les expériences.

### P1 — Réparer/diagnostiquer l’augmentation

Exécuter les ablations de corruption décrites dans la section 5 avec les mêmes splits, labels, features et évaluateur. Comparer les distributions de vitesse et les résultats par match/scénario.

Cette étape est la priorité technique numéro un : la corruption actuelle change directement la signification des features de mouvement.

### P2 — Auditer les labels et les épisodes courts

Vérifier :

- les clocks/frame IDs et offsets entre tracking et événements ;
- le mapping home/away et les axes du terrain ;
- la gestion de la mi-temps et des valeurs manquantes ;
- la continuité des tracks et le nombre de joueurs observés ;
- `BallStatus` et `BallPossession` séparément ;
- les 142 possessions courtes : données visibles/manquantes, transition d’équipe, trajectoire avant/après, événement associé.

Pour chaque épisode raté, attribuer autant que possible une cause : absence de signal, feature insuffisante, mauvaise probabilité, suppression par Viterbi, équipe incorrecte ou frontière décalée.

Si vous examinez les 142 épisodes des matchs déjà utilisés, traitez ces matchs comme du matériel de développement/diagnostic à partir de maintenant, pas comme un test intact.

Ne jamais fusionner un segment simplement parce qu’il est court.

### P3 — Réévaluer les variantes du modèle avec une corruption corrigée

Comparer sur exactement les mêmes folds :

- candidat déployé (status collectif + contrôle équipe plus riche) ;
- deux modèles binaires full-feature ;
- modèle trois classes ;
- nearest-player + smoothing ;
- chaque variante sans décodage et avec le même Viterbi.

Les résultats antérieurs suggèrent un avantage des deux modèles binaires sur le sweep de développement, mais la décision finale doit venir du benchmark corrigé.

### P4 — Diagnostiquer le décodage et les frontières

Après avoir réparé les signaux d’entrée :

- comparer les coûts de transition actuels, plus faibles et dépendants des preuves ;
- distinguer turnovers directs H/A des frontières arrêt/reprise ;
- inclure proximité et mouvement relatif du receveur, continuité après réception et probabilités brutes ;
- rapporter frontières matchées, manquées et supplémentaires ;
- suivre spécifiquement le rappel des possessions de 0,5–1,5 s.

Critère : améliorer le `spellF1` global sans faire exploser les possessions supplémentaires, tout en améliorant de façon crédible les épisodes courts.

### P5 — Évaluer les forks concurrents de façon équitable

Candidats identifiés : Blaze2fire, ScoreAdvise, SAM-Analytics-2027.

- Vérifier branche, commit, licence, dépendances, données et règles spécifiques.
- Reproduire d’abord l’expérience documentée originale si elle est exécutable.
- Puis réentraîner/adopter l’approche sur les mêmes folds et corruptions que notre benchmark.
- Distinguer reproduction originale et réimplémentation de l’idée.
- Ne pas utiliser de poids ayant vu le match d’évaluation ni des prédictions concurrentes comme vérité terrain.
- Deux forks reproductibles suffisent si le temps est limité.

### P6 — Vidéo seulement si les expériences la justifient

Tester d’abord la récupération du ballon sur un petit ensemble de fenêtres sélectionnées de manière reproductible : gaps longs, pertes réelles du tracking, transitions incertaines et replays potentiels. Mesurer couverture supplémentaire, faux positifs, alignement et gain downstream sur des matchs externes annotés.

Une détection du ballon ne prouve pas la possession. Le gain doit être démontré sur `spellF1` externe. Le broadcast vient après le tactical video, uniquement s’il apporte une couverture fiable et unique.

### P7 — Geler et soumettre

- Choisir une configuration sur les résultats externes, pas sur les cinq minutes publiques.
- Épingler les dépendances.
- Régénérer le CSV depuis un environnement propre.
- Vérifier exactement 199 319 lignes, les frame IDs attendus, l’ordre, les labels permis et l’absence de valeurs vides/colonnes parasites.
- Exécuter le validateur officiel.
- Vérifier une deuxième exécution identique.
- Garder les assets du match hors de Git.
- Finaliser README / rapport technique / instructions de reproduction.

---

## 8. Protocole de benchmark à respecter

- Splits par match entier, jamais par frames aléatoires.
- Utiliser les sept matchs DFL dans une validation croisée par match (outer GroupKFold) si c’est compatible avec les manifests actuels.
- Toute sélection de paramètres doit être imbriquée dans les données d’entraînement de chaque fold.
- Les matchs `J03WN1` et `J03WQQ` ont déjà été utilisés pour des analyses et décisions : ne plus les présenter comme un test totalement intact.
- Les répétitions de corruption (seeds) ne créent pas de nouveaux matchs indépendants.
- Les sept matchs ne sont pas sept environnements football totalement indépendants ; certaines équipes peuvent se répéter.
- Garder les entrées, masques, labels, folds et évaluateurs identiques entre méthodes.
- Rapporter moyenne, variation par match, résultats par scénario de corruption et incertitude appariée.
- Ne pas appliquer un seuil universel arbitraire (par exemple un gain fixe de 0,04) comme preuve automatique de supériorité.

### Métriques à rapporter

- `spellF1` : métrique principale.
- `macroF1` et F1 par classe.
- `D` precision, recall et F1.
- Live frames incorrectement supprimées.
- Retard de reprise après arrêt.
- Possessions manquées / supplémentaires.
- Transition F1.
- Frontières matchées, manquées et supplémentaires.
- Erreur temporelle des frontières matchées.
- Recall des possessions courtes (0,5–1,5 s) à tIoU 0,5.
- Résultats séparés pour les gaps ballon et les pertes de joueurs.

Un test oracle utilisant les vrais états Live/Dead peut être utile pour diagnostiquer le plafond du classifieur d’équipe, mais il s’agit d’un résultat diagnostic supérieur théorique : jamais d’un score candidat ni d’une entrée de production.

---

## 9. Fichiers et commandes utiles dans le dépôt

Fichiers indiqués comme existants dans l’état actuel du projet :

```text
POSSESSION.md
requirements-possession.txt
DISCOVERY.md

possession/
  __main__.py
  benchmark.py
  learn.py
  features.py

 data/possession-model/
  benchmark.json

probes/competitive-audit/
  audit.json   # présent dans le workspace d'origine, mais exclu de Git
```

Autres éléments mentionnés dans les rapports :

- `POSSESSION.md` : instructions pour obtenir les XML publics DFL épinglés.
- `requirements-possession.txt` : dépendances de l’environnement possession.
- `DISCOVERY.md` : historique des recherches/forks.
- `data/possession-model/benchmark.json` : benchmark enregistré.
- `possession/benchmark.py`, `learn.py`, `features.py` : benchmark, apprentissage et features.
- `possession/__main__.py` : point d’entrée CLI.

Commande de reproduction indiquée :

```bash
python -m pip install -r requirements-possession.txt
python -m possession --fit
```

Placer le bundle organisateur sous `data/` ou définir `MTAH_DATA_DIR`. Consulter `POSSESSION.md` pour le chemin exact des fichiers DFL et vérifier les arguments CLI dans `python -m possession --help` avant exécution. Le rapport indique que la génération produit `data/submission.csv`.

Exécuter ensuite le validateur officiel (retrouver sa commande exacte dans le dépôt / starter kit ; ne pas inventer son nom).

### ATTENTION : artefacts d’audit non versionnés

Le dossier `probes/competitive-audit/` est explicitement exclu de Git par son `.gitignore`. Les éléments suivants existent dans le workspace de l’auteur de l’audit, mais **ne sont pas nécessairement présents dans un clone** :

- scratch code ;
- sortie numérique et `audit.json` ;
- neuf frames extraites localement.

Demander à l’auteur de les transférer séparément ou régénérer ces artefacts. Ne pas supposer qu’ils sont récupérables depuis le remote.

---

## 10. Décisions provisoires à ne pas prendre trop tôt

- Ne pas abandonner DFL avant d’avoir corrigé l’augmentation et audité les labels.
- Ne pas remplacer LightGBM par un Transformer/LSTM/CRF simplement parce que le score court-spell est faible.
- Ne pas supprimer Viterbi : il améliore le score global rapporté. Diagnostiquer ses suppressions plutôt que le retirer d’emblée.
- Ne pas imposer de durée minimale et ne pas fusionner automatiquement les possessions courtes.
- Ne pas utiliser le broadcast sans gain mesuré et alignement fiable.
- Ne pas faire des Smart Tags la vérité terrain de `D`.
- Ne pas annoncer que 0,3401 prédit le score privé ou garantit un Top 20 / Top 5.
- Ne pas présenter les matchs déjà examinés comme des tests restés intacts.

---

## 11. Références principales

- Repository organisateur : https://github.com/metrica-sports/more-than-a-hack-2027
- Règles : https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md
- Documentation challenge / data / evaluation / submission : consulter les fichiers correspondants dans le repository organisateur.
- IDSSE / Sportec Open DFL : Bassek, Rein, Weber et al. (2025), *An integrated dataset of synchronized spatiotemporal event and tracking data in elite soccer* (référence citée par l’équipe ; vérifier la version et les champs dans la documentation du dataset).

---

## TL;DR pour la reprise

**Ne commencez pas par changer de modèle.** Figez le candidat et corrigez d’abord l’augmentation : le bruit indépendant ajouté frame par frame crée des vitesses artificielles et détruit une partie du signal Live/Dead. Ensuite, auditez les labels DFL et les possessions courtes pour localiser les erreurs avant/après Viterbi. Recomparez les deux modèles binaires, le modèle trois classes et le candidat actuel sur les mêmes splits et corruptions corrigées. Gardez le code, les seeds, les métriques et les artefacts reproductibles ; le score privé reste inconnu.
