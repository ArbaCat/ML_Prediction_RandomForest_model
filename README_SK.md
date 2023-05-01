**ML_Prediction_randomforest_model**
Toto je skript jazyka Python, ktorý vykonáva normalizáciu údajov a trénuje model strojového učenia na poskytnutej množine údajov. 
Konkrétne v tomto prípade tento model vypočíta elektrický výkon na základe už vopred známej množiny príkladových údajov.

**Požadované knižnice**: pandas(na manipuláciu s údajmi), scikit-learn(na strojové učenie)

**Transformácia údajov**: MinMaxScaler (na normalizáciu). Je to užitočné, keď data su roznorodne

**Opravy údajov:** Na opravy údajov používam funkcie -

data.drop_duplicates(inplace=True)
vymazanie opakovania v súbore údajov
data.dropna(inplace=True)
vymazanie prázdnych častí

Je to celkom jednoduché a základné. A použil som data.isnull(), aby som skontroloval, či sa proces skončil bez problémov.

**Výber dat pre učenie**: Ako cieľ som vybral re_m1_svor, pretože to je parameter, ktorý chceme predpovedať. A ostatné parametre som použil ako učebné údaje.

výber stĺpcov na učenie
features = ["tepl_m", "zraz_m", "tlak_m", "v_rh_m", "v_s_m", "v_rh_m_50", "v_rh_m_125", "vlhk_m"]
výber cieľového parametra(stĺpca)
target = "re_m1_svor

**Testovanie modelu**: Myslím, že scikit metrics je dobrý nástroj na testovanie, pretože je prehľadný a jasný. MSE, MAE a R2 Score sú základné a jasné metriky.
A na konci skriptu som urobil jeden jednoduchý a ľahký manuálny test.

**Ako to funguje:**

Skript načíta údaje z poskytnutého súboru CSV, odstráni všetky duplicitné alebo prázdne údaje a oddelí stĺpec s dátumom.

Potom sa údaje normalizujú pomocou MinMaxScaler a rozdelia sa na tréningovú a testovaciu množinu. 
Stĺpec dátum sa pridá späť do normalizovaných údajov 
(snažil som sa vyhnúť chybe s vynechaným časom vo finálnych údajoch (stáva sa to na jednom mojom počítači, neviem prečo)) 
(**P.S.**: Vytvoril som 2 modely, pretože nemôžem denormalizovať údaje späť (mám nejakú chybu s 2D/3D parametrami údajov, 
nemôžem tento problém na rychlo vyriešiť sám) 
(**P.S.S:** Prvý model (Full) normalizoval všetky parametre, druhý model (Full_model_re_m1_svor_DENORMALIZED) ponechal cieľový parameter denormalizovaný. 
Podľa metriky je prvý model oveľa lepší, ale kvôli problému s denormalizáciou som vytvoril druhý model. 
V podstate potrebujem len konzultáciu od niekoho skúsenejšieho ako ja aby som doladil prvy model).

Normalizované údaje sa uložia do nových súborov CSV a potom sa načítajú späť na trénovanie modelu. 
Skript vyberie príslušné funkcie a cieľový parameter (stĺpec) na trénovanie a testovanie.
Potom sa deklaruje RandomForestRegressor (použil som random_state=42, aby som si bol istý presnosťou a predvídateľnosťou modelu pri ďalšom použití 
(skúšal som štandardný parameter 22, ale po testovaní som zistil, že 42 funguje lepšie)) a vycvičí sa pomocou trénovacej množiny, 
potom sa otestuje na testovacej množine a vypíše sa score R2, stredná kvadratická chyba a stredná absolútna chyba.

Nakoniec len zadám do modelu normalizované údaje, ktoré už poznám, a model predpovie cieľovú hodnotu a vypíše výsledok,
ktorý som už poznal. Je to len na testovanie a model to robí dobre.
