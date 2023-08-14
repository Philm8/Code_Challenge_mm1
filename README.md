# Code_Challenge_mm1

""" Vorgehen im Data Science Projekt nach dem CRISP-DM"""

""" Phase 1: Business Understanding """

## Ist-Analyse des Betrachtungsumfeldes ##
# - Bikesharing-Unternehmen in Seoul möchte verstehen:
#   1. wie das Verhalten ihrer Kunden, durch die mit dem Klimawandel einhergehende sich verändernde Umwelt beeinflusst wird (sind verunsichert, ob da ein Zusammenhang besteht und das Kundenverhalten negativ beeinflusst wird)
#   2. wie sich das Kundenverhalten auf ihr Business (Anzahl an geteilten Fahrrändern) auswirkt
# - Kundendaten sind gegeben
# - Bericht ist gegeben, der zwei szenarienbehaftete Vorhersagen über die Änderungen des Klimas in Seoul trifft --> das Management ist unklar, wie dieses das Buisiness beeinflussen wird
# - Ich bin als Data Scientist eingestellt, um dem Unternehmen zu helfen ihre Daten zu verstehen

## Ableitung der konkreten Projektzeilstellung ##
# - C-Level Management(CEO, COO etc.) helfen zu entschieden auf zukünftige Schwangungen der Nachfrage ihrer Bikesharing-Angebots zu reagieren
# - datenbasierte Informationen darüber liefern, ob Seoul auch angesichts der erwarteten umweltbedingten Änderungen, bezogen auf die zukünftie Kunennachfrage einen attraktiven Markt darstellt

## Überführung des Kundenproblems in ein Lernproblem ##
# - Herausfinden, ob die Daten für die Entwicklung eines ML-Modells genutzt werden können, welches in der Lage ist den "rented bike count" vorherzusagen
# - Herausfinden, welche erklärenden Variablen/Einflussfaktoren am relevantesten für die Vorhersage sind

#%% Alle benötigten Packete reinladen
import chardet # Paket, um herauszufinden welche encodierung genutzt werden muss, um csv in pandas df umzuwandeln
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
from sklearn.model_selection import TimeSeriesSplit

print("Reinladen der notwendigen Python-Bibliotheken abgeschlossen!!!")



#%% Herauszufinden welche encodierung genutzt werden muss, um csv in pandas df umzuwandeln

csv = "C:/Users/portmann/Desktop/Uni/Bewerbungsunterlagen/MM1/Code Challenga/SeoulBikeData.csv"
with open(csv, 'rb') as Datei: # "rb" steht für read binary, um die Datei im Binärcode zu öffnen (dh. nur in Darstellung von 0 und 1 Kombinationen, welche die Zeichen in der Datei ergeben)
    result = chardet.detect(Datei.read())
print(result['encoding']) # --> liefert ISO-8859-1

print("Herauszufinden welche encodierung genutzt werden muss, um csv in pandas df umzuwandeln abgeschlossen!!!")





""" Phase 2: Data Understanding """

## Zusammensetzung des Datensatzes ##
# - 14 Attribute (davon 13 erklärenden variable & 1 Zielgröße (rented bike count))
# - Merkmalsträger ist Zeit (multivariates Merkmal, da p < 2 Merkmale pro Merkmalsträger erfasst werden)
#%% # Daten als pandas Dataframe reinladen
df = pd.read_csv(csv, encoding="ISO-8859-1")
print(df)
namen_attribute = df.columns
# Anzahlen der Beobachtungen & Attribute, sowie deren Namenausgeben
Anzahl_Beobachtungen, Anzahl_Attribute = df.shape
print("Anzahl der Zeilen/Beobachtungen:", Anzahl_Beobachtungen)
print("Anzahl der Spalten/Attribute:", Anzahl_Attribute)
print("Namen der Attribute:", namen_attribute)

print("Zusammensetzung des Datensatzes untersuchen abgeschlossen!!!")



#%% Datein in Excel-Datein umformatieren, zur zusäzlichen Sichprüfung bestimmter Aspekte
# Excel-Datei öffnen
with pd.ExcelWriter("Code_Challenge_Phillip.xlsx", engine="openpyxl", mode="a") as writer: # mode="a"für anhängen, also den neu befüllten Sheet an die Datei (hab so gemacht, damit Datei nicht voll wird, falls ich öfter die Zelle ausführe :)

    # Überprüfen ob Sheet bereits in der Excel-Datei existiert
    if "Datensatz_BikeSharing" in writer.book.sheetnames:

        # Sheet auswählen
        sheet = writer.book["Datensatz_BikeSharing"]

        # Sheet löschen
        writer.book.remove(sheet)

    # DataFrame in Excel schreiben
    df.to_excel(
        writer, sheet_name="Datensatz_BikeSharing", index=False) # index Nummern des dfs nicht als eine extra Spalte in Excle schreiben

print("df in Excel schreiben für zusätzliche Sichtkontrolle des Datensatzes abgeschlossen!!!")



## Bewertung der Datenreife ##
# - Orientierung an dem Reifegrandmodell von Eckelmann 2019
#   - Datenerfassung: keine Information vorliegen!
#   - Vollständigkeit: Erfassung der als relevant geltenden Merkmale --> Stufe 3

#   - Stichprobenumfang: große Stichprobe je Attribut & Attributsausprägung --> Stufe 3
#%% Stichprobenumfang prüfen je Attributsausprägung (je Attribut ist sie erstmal mit 8760 Beobachtungen gleich groß)
unique_Anzahl_pro_Attribut = df.nunique()
print("Anzahl der unterschiedlichen Ausprägungen je Attribut: ", unique_Anzahl_pro_Attribut)

# liefert Anzahl der Beobachtungsanzahl pro unterschiedliche Attributsausprägung
for Attribut in df.columns:
    print(df.columns) # object
    print(Attribut) # string
    Beobachtungsanzahl = df[Attribut].value_counts() # gibt Serie mit zwei Spalten aus (1. Spalte: eindeutige Ausprägunge, 2. Spalte: Anzahl wie oft die Asupräugng vorkommt)
    print(f"Attribut '{Attribut}':") # durch f string einbettung erscheint wert der variable bzw. attribut string anstelle von Attribut
    print(Beobachtungsanzahl)
    print()

print("Anzahl der Ausprägungen je Attribut & Stichprobenumfang je Attributsausprägung untersuchen abgeschlossen!!!")


#   - Datenhaltung: keine Information!!!
#   - Datenformat: Unterschiedliche , direkt überführbares Format (.csv) --> Stufe 3


#%% Zählen der metrischen und nominalen (kategorialen) Attribute
metrische_Attribute = df.select_dtypes(include=['number']).columns # liefert Array mit den Namen drin
nominale_Attribute = df.select_dtypes(include=['object']).columns
print(metrische_Attribute)
print(nominale_Attribute)

Anzahl_metrische_Attribute = len(metrische_Attribute)
Anzahl_nominale_Attribute = len(nominale_Attribute)

print(f"Anzahl der metrischen Spalten: {Anzahl_metrische_Attribute}")
print(f"Anzahl der nominalen Spalten: {Anzahl_nominale_Attribute}")

#   - Datenstruktur: Strukturierte gemischt skalierte Daten (metrisch --> Rented Bike Count, Temperatur((°C), Humidity(%)/Feuchtigkeit, Wind speed (m/s),
#                                                            Visibility (10m), Dew point temperature(°C)/Taupunkttemperatur(°C), Solar Radiation/Sonneneinstrahlung (MJ/m2), Rainfall(mm),
#                                                            Snowfall(cm); nominal --> Seasons, Holiday, Fuctioning Day/Werktag)) --> Stufe 3
print("Zählen der metrischen und nominalen (kategorialen) Attribute abgeschlossen!!!")


#   - Merkmalsausprägung: Aggregiert Istwerte oder Rodaten mit geringer Abtastrate (stundlich pro Tag) --> Stufe 3


#%% Konsistenz überprüfen mit ein paar Tests/Ausgaben und per Sichtprüfung in Excel
# Prüfe auf leere Einträge (fehlende Werte) im DataFrame
fehlende_Einträge = df.isna()

print("Leere Einträge im DataFrame:",(fehlende_Einträge))

Gesamtanzahl = df.isna().sum()
print("Gesamtanzahl der fehlenden Einträge im df pro Attribut:", Gesamtanzahl)
# --> keine fehlenden/leeren Einträge im df

# Prüfe ob Ausprägungsanzahl spezifischer Attribute im Kotext Sinn macht
unique_Anzahl_pro_Attribut = df.nunique()
print("Anzahl der unterschiedlichen Werte pro Spalte:",unique_Anzahl_pro_Attribut)
# date = 365 macht sinn für ein Jahr
# Hour = 24 macht sinn für Stundenanzahl an einem Tag
# Season = 4 macht sinn für 4 Jahreszeiten im Monat
# Functioning day (Werkstag) = 2 mit Ja oder Nein macht Sinn

#   - Konsistenz: Wenige logsiche Widersprüche (z.B. 13 Tage ohne Vermietung --> Geschäft geschlossen?;  4300 mal ist Solar Radiation bei 0 --> so dichte Wolkendecke? eig. kommt Sonne dennoch bisschen durch) --> Stufe 3

#   - Rückverfolgbarkeit: schwer zu beurteilen, da nur eine Datei aber darauf bezogen 

print("Konsistenz überprüfen mit ein paar Tests/Ausgaben und per Sichtprüfung in Excel abgschlossen!!!")



## Deskriptive und explorative Analyse der Daten ##
#%% Verteilung der Daten untersuchen
five_num_sum = df.describe()
print(five_num_sum)

# durch die Spalten des dfs iterieren
for Attribut in df.columns:
    if df[Attribut].dtype == 'object':
        # Filtere nur kategoriale Spalten/Attribute (Typ 'object')
        ausprägungs_anzahl = df[Attribut].value_counts()
        
        # Erstelle ein Stabdiagramm für die kategorialen Attribute
        plt.figure(figsize=(8, 6))  
        ausprägungs_anzahl.plot(kind='bar')
        plt.title(f"Stabdiagramm für Attribut '{Attribut}'")
        plt.xlabel("Ausprägung")
        plt.ylabel("Anzahl")
        plt.show()
    
    else:
        # nur metrische Spalten/Attribute (nicht kategorial) filtern
        data_range = df[Attribut].max() - df[Attribut].min()
        #print(data_range)
        bin_width = 1.0  # Die gewünschte Breite der Balken im Histogramm, da Höhe = Fläche durch Breite, damit Hähe quasi genau die Häufigkeit der Ausprägung ist ist
        
        # Berechne die Anzahl der Bins basierend auf der Breite
        num_bins = int(data_range / bin_width)
        #print(num_bins)
        
        plt.figure(figsize=(8, 6))
        plt.hist(df[Attribut], bins=num_bins)
        #plt.hist(df[Attribut])
        plt.title(f"Histogramm für Attribut '{Attribut}' (Breite = 1)")
        #plt.title(f"Histogramm für Attribut '{Attribut}'")
        plt.xlabel("Wert")
        plt.ylabel("Anzahl")
        plt.show()



# %% Zusammenhangsanalyse zwischen den Attributen ##
# Streudiagramme anstelle des Korelationskoeffizienten, da es kein Informationsverlust hinsichtlich der Art des Zusmamenhangs (linear, quadratisch etc. gibt)
# MERKEN: Der Korrelationskoeffizient misst lediglich die Stärke und Richtung der linearen Beziehung zwischen zwei Variablen. 
# Ein hoher Korrelationskoeffizient (z.B. 0,82) zeigt an, dass es eine starke lineare Beziehung zwischen den Variablen gibt, aber er sagt nichts über die Form dieser Beziehung aus.

#Es ist durchaus denkbar, dass zwei Variablen eine nicht-lineare Beziehung haben, wie zum Beispiel eine exponentielle, quadratische oder logarithmische Beziehung, und dennoch einen hohen Korrelationskoeffizienten aufweisen
# Berechne die Korrelationsmatrix für numerische Attribute
df_metric_attributes = df.select_dtypes(include=['float64', 'int64'])
print(df_metric_attributes)
correlation_matrix = df_metric_attributes.corr()

# Plot der Korrelationsmatrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Korrelationsmatrix der metrischen Attribute")
plt.show()

# Streudiagramme-Matrix erstellen
sns.set(style="ticks")
sns.pairplot(df, kind="scatter")
plt.show()

# Fazit: Wichtige Bewertungen des linearen Zusammenhangs in Bezug auf die Zielgröße
# - Rented Bike Count weist eine ...
#   - schwach positiv korrelierten linearen Zusammenhang zu Hour auf
#   - mittlel starker positiv korrelierten linearen Zusammenhang zu Temperbereich auf
#   - schwach negativ korrelierten linearen Zusammenhang zu Humidity auf
#   - schwach positiv korrelierten linearen Zusammenhang zu Wind speed auf
#   - schwach positiv korrelierten linearen Zusammenhang zu Visibillty auf
#   - schwach positiv korrelierten linearen Zusammenhang zu Dew point temperature auf
#   - schwach positiv korrelierten linearen Zusammenhang zu Solar Radiation auf
#   - schwach negativ korrelierten linearen Zusammenhang zu Rainfall auf
#   - schwach negativ korrelierten linearen Zusammenhang zu Snowfall auf
#
# - weitere Auffäligkeiten
#   - starker positiver linearer Zusammenhang wzischen Temperatur & Dew point temperature
#   - mittel starker positiver linearer Zusammenhang zwischen Dew point temperature & Humidity
#   - mittel starker negativer linearer Zusammenhang zwischen Vsibility & Humidity
#   - möglicher negativer quadratischer Zusammenhang zwischen Hour & Solar Energy
#   - möglicher negativer exponentieller Zusammenhang zwischen Renting count & rainfall
#   - möglicher negativer exponentieller Zusammenhang zwischen Renting count & snowfall
print("Zusammenhangsanalyse zwischen den metrischen Attributen abgeschlossen!!!")



#%% Untersuchung des Einflusses der nominalen Attribute auf die Zielgröße
metric_attributes = df.select_dtypes(include=[np.number]).columns
nominal_attributes = df.select_dtypes(include=['object']).columns
nominal_attributes = [attr for attr in nominal_attributes if attr != 'Date'] # zieht Date Attribut raus

# Punktdiagramme erstellen
sns.set(style="whitegrid")

for metric_attr in metric_attributes:
    for nominal_attr in nominal_attributes:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=nominal_attr, y=metric_attr, hue=nominal_attr, palette='Set1')
        plt.title(f"Scatter Plot: {metric_attr} vs. {nominal_attr}")
        plt.show()
        
# Fazit: Wichtige Bewertungen des Einflusses der nominalen Attribute auf die Zielgröße
# - Einfluss auf Rented Bike Count:
#   - Seasons --> Summer > Spring = Automn > Winter
#   - Holiday --> No Holiday > Holiday
#   - Functioning Day --> Yes > No
#   - 
print("Zusammenhangsanalyse zwischen den nominalen Attributen abgeschlossen!!!")




# %% Aureißer untersuchen
# Boxplots für jedes Attribut erstellen in gemeinsamen Plot
plt.figure(figsize=(10, 6))  
df.boxplot()
plt.title("Boxplots für jedes Attribut")
plt.xticks(rotation=45)  # Attributnamen auf der x-Achse um 45 Grad rotieren
plt.ylabel("Wert")
plt.xlabel("Attribut")
plt.show()

# Anzahl der Spalten/Attribute im DataFrame
num_columns = len(df.columns) #Anzahl der Attribute im df
num_cols = 3 # gewünschte Anzahl der Spalten in der Figure

# Berechnung der Anzahl der Zeilen und Spalten für die Figure
num_rows = math.ceil(num_columns / num_cols) # ceil um Anzahl der Zeilen in Figure aufzurunden

# Figure und Achsen erstellen
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 20))
fig.subplots_adjust(hspace=0.5)  # Platz machen zwischen den Subplots (vertikal)

# Iteriere durch die Spalten des DataFrames und erstelle die Boxplots für alle metrischen Attribute
for idx, Attribut in enumerate(df.columns): # enumerte ermöglicht in Funktion idx als Zähler und Attribut als string zu brücksichtigen
    print(idx)
    print(Attribut)
    if df[Attribut].dtype == 'object':
        print("kein metrisches Attribut")
    
    else: 
        
        row_idx = idx // num_cols # gibt die ganze Zahl des Quotienn zurück d.h. bei 3 2
        print(row_idx)
        col_idx = idx % num_cols # Rest der Division
        print(col_idx)
        ax = axes[row_idx, col_idx] # wählt Achsenobjekt für akutellen Subplot aus dem Array der Achses-Objekte aus
        print(ax)
        
        # Erstelle den Boxplot für die aktuelle Spalte
        df.boxplot(column=Attribut, ax=ax)
        ax.set_title(Attribut)  # Setze den Titel der Boxplot-Achse
        
        # Stat. Kennzahlen für den Boxplot
        median = df[Attribut].median()
        q1 = df[Attribut].quantile(0.25)
        q3 = df[Attribut].quantile(0.75)
        minimum = df[Attribut].min()
        maximum = df[Attribut].max()
        iqr = q3-q1
        
        # Zeichne Linien für die stat. Kennzahlen
        ax.axhline(y=median, color='r', linestyle='--', label='Median')
        ax.axhline(y=q1, color='g', linestyle='--', label='Q1')
        ax.axhline(y=q3, color='b', linestyle='--', label='Q3')
        ax.axhline(y=minimum, color='y', linestyle='--', label='Min')
        ax.axhline(y=maximum, color='m', linestyle='--', label='Max')

        #lower_whisker = q1 - 1.5 * iqr
        #upper_whisker = q3 + 1.5 * iqr
        
        # Zeichne Linien für die Whisker-Grenzen
        #ax.axhline(y=lower_whisker, color='y', linestyle='--', label='Unterer Whisker')
        #ax.axhline(y=upper_whisker, color='c', linestyle='--', label='Oberer Whisker')

        # Werte der Whisker-Grenzen als Text hinzufügen
        #ax.text(1.05, lower_whisker, f'Lower: {lower_whisker:.2f}', color='y', transform=ax.get_yaxis_transform())
        #ax.text(1.05, upper_whisker, f'Upper: {upper_whisker:.2f}', color='c', transform=ax.get_yaxis_transform())

        ax.set_title(Attribut)  
        ax.legend() 
        ax.legend(bbox_to_anchor=(1, 0.75)) # bbox_to_anchor=(1, 0.75): Die Legende wird 1 Einheit von der rechten Kante des Plots entfernt und auf halber Höhe der Figur vertikal zentriert (0,0) --> Start in der unteren linken Ecke des Axes-Objekt (ein Subplot) und geht bis (1,1) in der oberen rechten Ecke
        
plt.tight_layout() # Platz optimieren 
plt.show() 

#Fazit:
# - Attribute (Rented Bike Count, Wind speed, Solar Radiation, Rainfall & Snowfall haben viele Ausreißer oberhalb des obrenen Whisker, also reißen zu größeren Werte aus)
# - den nominalen Attribute bringt die Visualisierung per Boxplot keinen Mehrwert
# - ML-Modell mit Entscheidungsbäumen sind weniger anfälliger für Aureißer
# - Auswirkungen der Ausreißer auf das Modell untersuchen, also mal mit ma

print("Aureißer anhand Boxplots untersuchen abgeschlossen!!!")



#%% Nochmal für die Präsi ausgewählte Attribute plotten !!!#

ausgewählte_attribute = df[["Rented Bike Count","Wind speed (m/s)", "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)"]]
print(ausgewählte_attribute)
print(ausgewählte_attribute.columns) # .columns ohne klammer damit, spaltenamen gegeben werden

# Anzahl der Spalten/Attribute im DataFrame
num_columns = len(ausgewählte_attribute.columns) #Anzahl der Attribute im df
print(num_columns)
num_cols = 5 # gewünschte Anzahl der Spalten in der Figure

# Figure und Achsen erstellen
fig, axes = plt.subplots(ncols=num_cols, figsize=(15, 5))
fig.subplots_adjust(hspace=0.5)  # Platz machen zwischen den Subplots (vertikal)

# Iteriere durch die Spalten des DataFrames und erstelle die Boxplots für alle metrischen Attribute
for idx, Attribut in enumerate(ausgewählte_attribute.columns): # enumerte ermöglicht in Funktion idx als Zähler und Attribut als string zu brücksichtigen
    print(idx)
    print(Attribut)
    if df[Attribut].dtype == 'object':
        print("kein metrisches Attribut")
    
    else: 
        
        col_idx = idx % num_cols # Rest der Division
        print(col_idx)
        ax = axes[col_idx] # wählt Achsenobjekt für akutellen Subplot aus dem Array der Achses-Objekte aus
        print(ax)
        
        # Erstelle den Boxplot für die aktuelle Spalte
        df.boxplot(column=Attribut, ax=ax)
        ax.set_title(Attribut)  # Setze den Titel der Boxplot-Achse
        
        # Stat. Kennzahlen für den Boxplot
        median = df[Attribut].median()
        q1 = df[Attribut].quantile(0.25)
        q3 = df[Attribut].quantile(0.75)
        minimum = df[Attribut].min()
        maximum = df[Attribut].max()
        iqr = q3-q1
        
        # Zeichne Linien für die stat. Kennzahlen
        ax.axhline(y=median, color='r', linestyle='--', label='Median')
        ax.axhline(y=q1, color='g', linestyle='--', label='Q1')
        ax.axhline(y=q3, color='b', linestyle='--', label='Q3')
        ax.axhline(y=minimum, color='y', linestyle='--', label='Min')
        ax.axhline(y=maximum, color='m', linestyle='--', label='Max')

        ax.set_title(Attribut)  
        ax.legend() 
        ax.legend(bbox_to_anchor=(1, 0.75)) # bbox_to_anchor=(1, 0.75): Die Legende wird 1 Einheit von der rechten Kante des Plots entfernt und auf halber Höhe der Figur vertikal zentriert (0,0) --> Start in der unteren linken Ecke des Axes-Objekt (ein Subplot) und geht bis (1,1) in der oberen rechten Ecke
        
plt.tight_layout() # Platz optimieren 
plt.show() 

print("Aureißer anhand Boxplots für Präsi plotten abgeschlossen!!!")


# %% Untersuchung des Einflusses des Attributes Date auf die metrischen Größen

df_date = df.copy() # Kopieerstellung, damit df nicht immer beim nochmal alufen der Zelle ebenfalls überschrieben wird
df_date.set_index('Date', inplace=True)# Index auf das Datum setzen
print(df_date)
df_date.index = pd.to_datetime(df_date.index, format='%d/%m/%Y')
print(df_date)

ausgewählte_attribute = df[["Rented Bike Count","Wind speed (m/s)", "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)"]]
print(ausgewählte_attribute)
print(ausgewählte_attribute.columns) # .columns ohne klammer damit, spaltenamen gegeben werden


# Anzahl der Spalten/Attribute im DataFrame
num_columns = len(ausgewählte_attribute.columns) #Anzahl der Attribute im df
print(num_columns)
num_cols = 5 # gewünschte Anzahl der Spalten in der Figure

# Figure und Achsen erstellen
fig, axes = plt.subplots(ncols=num_cols, figsize=(15, 5))
fig.subplots_adjust(hspace=0.5)  # Platz machen zwischen den Subplots (vertikal)

# Schleife durch die metrischen Spalten und Plotten gegen das Datum
for idx, Attribut in enumerate(ausgewählte_attribute.columns):
    
    col_idx = idx % num_cols # Rest der Division
    print(col_idx)
    ax = axes[col_idx] # wählt Achsenobjekt für akutellen Subplot aus dem Array der Achses-Objekte aus
    print(ax)
    
    ax.plot(df_date.index, df_date[Attribut])
    ax.set_xlabel('Date')
    ax.set_ylabel(Attribut)
    
    # X-Achse formatieren, mehr Datumsangaben anzeigen
    num_display_points = 3  # Anzahl der angezeigten Datenpunkte auf der X-Achse
    step = max(len(df_date) // num_display_points, 1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xticks(df_date.index[::step])
    
    
plt.tight_layout()
plt.show()

print("Untersuchung und Plotten des Einflusses des Attributes Date auf die metrischen Größen")

# Fazit: Untersuchung des Einflusses des Attributes Date auf die metrischen Größen
# Zielgröße Rentent bike counts:
#                               - Ende Juni Anfang August Höchstphase
#                               - Anfang Februar 2017 Tiefstphase
#                               - Ende November/Anfang Dezember 2018 steigt verdoppelt sich Anzahl der gemieteten Fahrräder fast zum Vorjahr 2017 --> wachsendes Business





""" Phase 3: Data Preparation (Konstruktion des finalen Datensatzes für die Modellierng)"""
#%% Skalierung der metrischen Attribute:
# # Ist wichtig, damit verhindert wird, dass bestimmte Attribute aufgrund ihrer unterschiedlichen Skalen eine übermäßige Gewichtung im Modell haben
# # Min-Max-Skalierung
# print(df)
# date_attributes = ['Date']
# target = ['Rented Bike Count']
# metric_attributes = ['Hour', 'Temperature(°C)', 'Humidity(%)',
#        'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
#        'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']
# nominal_attributes = ['Seasons','Holiday', 'Functioning Day']

# scaler = MinMaxScaler()
# print(df[target])
# print(df[date_attributes])
# print(df[metric_attributes])
# print(df[nominal_attributes])
# scaled_data = scaler.fit_transform(df[metric_attributes])
# print(scaled_data)

# # Erzeugung eines neuen skalierten DataFrames
# df_metric_attributes_scaled = pd.DataFrame(scaled_data, columns=metric_attributes)
# print(df_metric_attributes_scaled)

# df_scaled_merged = pd.concat([df[['Rented Bike Count','Date','Seasons','Holiday', 'Functioning Day']], df_metric_attributes_scaled[metric_attributes]], axis=1) # axis = 1 heißt dfs horizontal entlang der Spalten zusamen fügen
# print(df_scaled_merged)

# #%% Kodierung nominale Attribute:
# # Nominale Attribute müssen in numerische Form gebracht werden, damit sie von maschinellen Lernalgorithmen verarbeitet werden können.
# # One-Hit-Encoding wählen, damit implizite Anordnung der Klassen ML-Modell nicht möglicherweise beeinflusst
# # One-Hit-Encoding = jede Asuprägung eines nominalen Attributs in eigene Spalte mit binären Werten (0 oder 1) umwandeln
# print(df_scaled_merged)
# df_scaled_merged_encoded = pd.get_dummies(df_scaled_merged, columns=['Seasons', 'Holiday', 'Functioning Day'], drop_first=True)
# print(df_scaled_merged_encoded)
# df_scaled_merged_encoded[['Seasons_Spring','Seasons_Summer','Seasons_Winter',
#        'Holiday_No Holiday', 'Functioning Day_Yes']] = df_scaled_merged_encoded[['Seasons_Spring','Seasons_Summer','Seasons_Winter',
#        'Holiday_No Holiday', 'Functioning Day_Yes']].astype(int) # damit True und False als numerische Werte 0 & 1 im df_encoded angezeigt werde
# print(df_scaled_merged_encoded)

# df_scaled_merged_encoded.set_index(['Date'], inplace=True)
# df_scaled_merged_encoded_date = df_scaled_merged_encoded
# print(df_scaled_merged_encoded_date)

#unskalierters Vorgehen weil XGBoost nehmen

#%% Kodierung nominale Attribute:
# Nominale Attribute müssen in numerische Form gebracht werden, damit sie von maschinellen Lernalgorithmen verarbeitet werden können.
# One-Hit-Encoding wählen, damit implizite Anordnung der Klassen ML-Modell nicht möglicherweise beeinflusst
# One-Hit-Encoding = jede Asuprägung eines nominalen Attributs in eigene Spalte mit binären Werten (0 oder 1) umwandeln
print(df)
df_encoded = pd.get_dummies(df, columns=['Seasons', 'Holiday', 'Functioning Day'], drop_first=True)
print(df_encoded)
df_encoded[['Seasons_Spring','Seasons_Summer','Seasons_Winter',
       'Holiday_No Holiday', 'Functioning Day_Yes']] = df_encoded[['Seasons_Spring','Seasons_Summer','Seasons_Winter',
       'Holiday_No Holiday', 'Functioning Day_Yes']].astype(int) # damit True und False als numerische Werte 0 & 1 im df_encoded angezeigt werde
print(df_encoded)

df_encoded['Date'] = pd.to_datetime(df_encoded['Date'], format='%d/%m/%Y')
df_encoded['Hour'] = pd.to_datetime(df_encoded['Hour'], format='%H').dt.time  # Hour in Uhrzeiten umwandeln
df_encoded['Hour'] = df_encoded['Hour'].apply(lambda x: x.strftime('%H:%M'))  # Uhrzeiten in String-Format umwandeln
df_encoded.set_index(pd.to_datetime(df_encoded['Date'].astype(str) + ' ' + df_encoded['Hour'], format='%Y-%m-%d %H:%M'), inplace=True)

# Spalten "Date" und "Hour" entfernen
df_encoded.drop(['Date', 'Hour'], axis=1, inplace=True)
print(df_encoded)

print("Kodierung nominale Attribute und Phase Data Preparation abgeschlossen!!!")



#%% Kodiertes df in Excel-Datei umformatieren, zur zusäzlichen Sichprüfung bestimmter Aspekte
# Excel-Datei öffnen
with pd.ExcelWriter("Code_Challenge_Phillip.xlsx", engine="openpyxl", mode="a") as writer: # mode="a"für anhängen, also den neu befüllten Sheet an die Datei (hab so gemacht, damit Datei nicht voll wird, falls ich öfter die Zelle ausführe :)

    # Überprüfen ob Sheet bereits in der Excel-Datei existiert
    if "Datensatz_kodiert" in writer.book.sheetnames:

        # Sheet auswählen
        sheet = writer.book["Datensatz_kodiert"]

        # Sheet löschen
        writer.book.remove(sheet)

    # DataFrame in Excel schreiben
    df_encoded.to_excel(
        writer, sheet_name="Datensatz_kodiert", index=True) # index Nummern des dfs nicht als eine extra Spalte in Excle schreiben

print("df_edcoeded in Excel schreiben für zusätzliche Sichtkontrolle des Datensatzes abgeschlossen!!!")





""" Phase 4: Modeling """
## Auswahl des Data Mining Verfahrens ##
# Aufgabe aus gegebenen Attributswerten eine unbekannte Zielgröße zu bestimmen
# dafür werden Daten bereit gestellt, bei denen das Label bekannt ist (also in diesem Fall der Rented Bike Count) --> Überwachtes Lernverfahren
# Zielgröße bzw. Label ist metrisch --> Regression
# Wahl des Algorithmus aus Erfahrung --> XGBoost
# Gründe für XGBoost:
#                    - bietet Funktion die Feature Importance zu interpretieren
#                    - XGBoost ist robust gegenüber unterschiedlichen Skalen der Eingabewerte und kann gut mit unskalierten Daten umgehen (arbeitet mit Entschiedungsbäumen & wird nicht von linearen Abhängigkeiten zwischen Attributen beeinflusst)


#%% Training des Modells --> Da große Datenmenge verfügbar nur Aufteilung des Datensatzes in Traingsdaten (80%) & Testdaten (20%) (sollte zuverlässige Ergebnisse liefern & Risko von Überanpassung geringer!)
# Aufteilung in Attribute (X) und Zielgröße (y)
x = df_encoded.drop('Rented Bike Count', axis=1)
y = df_encoded['Rented Bike Count']
print(x)
print(y)
# Daten in Trainings- und Testsets aufteilen (zeitliche Aufteilung)
train_size = int(0.8 * len(df_encoded))  # 80% als Trainingsdaten, 20% als Testdaten
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(x_train, x_test)
print(y_train, y_test)

# XGBoost-Modell erstellen und trainieren
model = xgb.XGBRegressor(n_estimators=1000,       # Anzahl der Bäume
    learning_rate=0.1,       # Lernrateteuert, wie stark jeder Entscheidungsbaum die Vorhersage des vorherigen Baums korrigiert
    max_depth=6,             # Maximale Baumtiefe wie tief die Entscheidungsbäume im Ensemble wachsen dürfen. 
    min_child_weight=1,      # Minimale Summe der Instanzgewichte (wie viele Instanzen/Datenpunkte in einem Blattknoten landen müssen, damit es gebildet wird) in Blattknoten
    subsample=0.8,           # Bruchteil der Trainingsdaten für jeden Baum
    colsample_bytree=0.8,    # Bruchteil der Merkmale für jeden Baum
    objective='reg:squarederror',  # Zielfunktion für Regression
    eval_metric='rmse',      # Bewertungsmetrik
    early_stopping_rounds=10,  # Frühzeitiges Beenden bei fehlender Verbesserung
    random_state=42          # Zufallsgenerator für Wiederholbarkeit des Trainings
)

model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=True) # erwendung von eval_set ermöglicht den Fortschritt des Modells während des Trainingsprozesses zu überwachen, indem du sowohl die Leistung auf den Trainingsdaten als auch auf den Testdaten beobachtet wird --> Überanpassung wird erkannt & dann abgeborchen! --> eval_score wird nach jeder Iteration ausgegeben!
# n diesem Beispiel wird das Training gestoppt, wenn die Leistung auf den Testdaten (x_test, y_test) für 10 aufeinanderfolgende Runden nicht mehr verbessert wird

# Feature Importance ermitteln (gibt nur klare Einsicht, wenn Features nicht stark miteinander korrelieren (Tautemperatur trotzdem mitreinnehmen, da Unternehmen Bewetung des Einflusses aller Daten auf Zielgröße wünscht)!!! --> ist bei uns der Fall also nutzbar!)
df_feature_importance = pd.DataFrame(data=model.feature_importances_, index=model.feature_names_in_, columns=["importance"])
print(df_feature_importance)
df_feature_importance.sort_values("importance").plot(kind="barh", title ="Feature importance")
plt.show()
print(df_feature_importance)



#%% Vorhersagen machen auf den Testdaten
y_pred = model.predict(x_test)
print(y_pred)

# Index in eine numerische Liste umwandeln
index_range = list(range(len(y_test)))

# Plot der Vorhersagen und der richtigen Werte
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='True Values')
plt.plot(y_test.index, y_pred, label='Predictions', linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Rented Bike Count')
plt.title('True Values vs. Predictions')
plt.legend()
plt.show()

# Modellleistung evaluieren (MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # gleiches Fehlermaß, welches auch beim Trainings zum vergleichen angeschaut wurde
print(f"Mean Squared Error auf dem Testset: {mse:0.2f}") # 0.2f, damit nur zwei NAchkommastellen angezeigt werden
print(f"Root Mean Squared Error auf dem Testet: {rmse:0.2f}")

# Untersuchung der absoluten Abweichung zwischen Test- & Pred Daten
diff_Betrag = np.abs(y_test-y_pred)
print(y_test)
print(y_pred)
print(diff_Betrag)

# nach Reihenfolge sortieren, um um zu gucken an welchen Daten besonder gute Vorhersagen geschehen sind
print(diff_Betrag.sort_values())

# Fazit: Modellerstellung
#       - Trend mit Ups & Downs erkennt das Modell gut, nur schätzt den Rented Bike Count i.d.R. etwas niedriger aus, als er eigentlich ist ---> pessimistische Vorhersage
#       - beste Vorhersagen auf mit geringster Abweichung im Zeitraum (Ende September - Anfang Oktober)
#       - schlechteste Vorhersagen mit größter Abweichung im Zeitraum (Mitte Oktober bis Mitte November)
#       - Modell zeigt Tendenzen RMSE mit 460 aber viel zu hoch, Wert Nahe Null wäre perfekt.




""" Phase 4: Evaluation """
#%% Szenario 1: Durchschnittstemperatur erhöht sich um 2 Grad Celsius
#           - 1. Schritt: Druchschnittstemperatur der ganzen Daten berechnen
#           - 2. Schirtt: Durchschnittstemperatur durch Anzahl der Beobachtungen teilen, um darauf zu kommen wieviel pro Beobachtung hinzukommt
#           - 3. Schritt: Dann den berechneten Wert auf die Temperatur jeder Beobachtung drauf rechnen 
print(x_test)
wert_zu_addieren = 2 # Wert, der hinzugefügt werden soll

# Füge den Wert zu jeder Zeile in der Spalte hinzu
x_test_szenario1 = x_test.copy() # copy, damit die dfs unabhängig voneinander sind und sich nicht gegenseitig überschreiben
x_test_szenario1['Temperature(°C)'] = x_test['Temperature(°C)'] + wert_zu_addieren
print(x_test_szenario1)

# Vorhersagen machen
print(y_pred)
y_pred_szenario1 = model.predict(x_test_szenario1)
print(y_pred_szenario1)

# Plot der Vorhersagen ohne Temperaturerhögung und mit
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_pred, label='Prediktions')
plt.plot(y_test.index, y_pred_szenario1, label='Predictions 2 Grad Temperaturerhögung', linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Rented Bike Count')
plt.title('Predictions alt vs. 2 Grad Temperaturerhögung')
plt.legend()
plt.show()

# Summme der vorhergesagten Rented Bike Counts übers Jahr
sum_pred = sum(y_pred)
sum_pred_szenario1 = sum(y_pred_szenario1)
differenz_rented_bike_count_szenario1 = sum_pred_szenario1-sum_pred
print("Summme der vorhergesagten Rented Bike Counts übers Jahr ohne Temperaturerhöhung:", sum_pred)
print("Summme der vorhergesagten Rented Bike Counts übers Jahr mit 2 Grad Temperaturerhöhung:", sum_pred_szenario1)
print("Anzahl der mehr gerenteten Bikes aufgrund der Temperaturerhögung: ", differenz_rented_bike_count_szenario1)

# Fazit:
#       - Steigerung der Temperatur hat positiven Einfluss --> mehr Fahrräder werden gemietet!!
#%% Szenario 2: Durchschnittstemperatur erhöht sich um 2 Grad Celsius
#           - 1. Schritt: Druchschnittsluftfeuchtigkeit der ganzen Daten berechnen
#           - 2. Schirtt: Druchschnittsluftfeuchtigkeit durch Anzahl der Beobachtungen teilen, um darauf zu kommen wieviel pro Beobachtung hinzukommt
#           - 3. Schritt: Dann den berechneten Wert auf die Luftfeuchtigkeit jeder Beobachtung drauf rechnen 
print(x_test)
wert_zu_addieren = 3 # Wert, der hinzugefügt werden soll

# Füge den Wert zu jeder Zeile in der Spalte hinzu
x_test_szenario2 = x_test.copy() # copy, damit die dfs unabhängig voneinander sind und sich nicht gegenseitig überschreiben
x_test_szenario2['Humidity(%)'] = x_test['Humidity(%)'] + wert_zu_addieren
print(x_test_szenario2)

# Vorhersagen machen
print(y_pred)
y_pred_szenario2 = model.predict(x_test_szenario2)
print(y_pred_szenario2)

# Plot der Vorhersagen ohne Temperaturerhögung und mit
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_pred, label='Prediktions')
plt.plot(y_test.index, y_pred_szenario2, label='Predictions 3 Prozentpunkte Luftfeuchtigkeiterhögung', linestyle='dotted')
plt.xlabel('Time')
plt.ylabel('Rented Bike Count')
plt.title('Predictions alt vs. 3 Prozentpunkte Luftfeuchitgkeitserhöhung')
plt.legend()
plt.show()
# Summme der vorhergesagten Rented Bike Counts übers Jahr
sum_pred = sum(y_pred)
sum_pred_szenario2 = sum(y_pred_szenario2)
differenz_rented_bike_count_szenario2 = sum_pred_szenario2-sum_pred
print("Summme der vorhergesagten Rented Bike Counts übers Jahr ohne Luftfeuchitgkeiterhöhung:", sum_pred)
print("Summme der vorhergesagten Rented Bike Counts übers Jahr mit 3 Prozentpunkte Luftfeuchitgkeitserhöhung:", sum_pred_szenario2)
print("Anzahl der mehr gerenteten Bikes aufgrund der Luftfeuchitgkeitserhöhung: ", differenz_rented_bike_count_szenario2)

#Fazit:
#       - Steigerung der Luftfeuchtigkeit hat negativen Einfluss --> weniger Fahrräder werden gemietet!!!

#%% Business beurteilen - Berechnung der Durchschnittszunahme anhand prozentualer Gewichtung
Gewichtete_Durchschnittszunahme = (0.70 * differenz_rented_bike_count_szenario1) + (0.30 * differenz_rented_bike_count_szenario2)
print("Gewichtete_Durchschnittszunahme unter Berücksichtigung der Prozente: ", Gewichtete_Durchschnittszunahme)
# Fazit:
#       - Manager können beruhigt werden, diese Entwicklung wird das Business positiv beeinflussen!!!
# %%

