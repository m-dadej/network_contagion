Wnioski:

- im wyzsza awersja do ryzyka tym wiecej bankructw oraz nizsze adequacy ratio. Chyba wynika to z tego ze parametry modelu bardziej motywuja do wyboru n zamiast l, poniewaz 1) lgd jest wyzsze -> ryzyko l jest wyzsze, 2) r_n ~ U(0.06, 0.07) ma niska wariancje ktora potem jest rowniez inputem do wyceniania n. Oba te parametry sa inne niz w paperze iniaki moze warto to zmienic tak aby zaleznosc miedyz awersja a capitla ratio i bankructwami byla sensowan

- wyzszy poziom risk aversion ma mniejszy niz spodziewany (o ile wgl ma) efekt na systemic risk. Wynika to z tego ze wyzszy risk aversion zmniejsza ilosc chetnych na borrowing i tym samym zmniejsza stope rownowagi, tym samym zwiekszajac zyskownosc inwestycji ktora offsetuje wyzsze risk aversion.

to do:

- sprawdz jeszcze raz czy na pewno mathcing algo daje bardziej ryzykowne pozycje bardziej sklonnym do ryzyaka bankom a nie na odwrot?

- porpawic warunek skonczenia petli: min. poprawa od poprzedniego najlepszego wyniku a nie od poprzedniego wyniku
- dodaj komentarze
- przepisz kod i dodaj wystandaryzowane znaki unicode
- chyba inaczej trzeba okreslic max_exposure w fund_mathcing. Obczaj jakie sa regulacje wdg BUS
    - sprawdz tez czy wgl to ograniczenie jest binding jakkolwiek

- optim_vars i A_ib pokazuja to samo a trzeba je niby zmieniac ciagle - mozna by ktores usunac
- zmienna "e" chyba nie jest potrzebna, zwyczajnie licz bilans i różnica będzie wynosiła e, dzieki temu nie bedziesz musial jej zmieniac zawsze

pytania:

- imbalance adjustment jesli za duzo l to dodaje borrowing i cash. Czy ma dodawac wszystkim borrowing czy tylko tym co maja jakikolwiek borrowing?

Problem:

- Oba U(E p) rozroznia meidzy malymi i duzymi bankami
- czasem obniżka r_l prowadzi do kontrintuicyjnego zwiekszenia sie imbalance. 
- chyba czasem moze sie zdazyc ze imbalance adjustment nie zadziala
- rzadko ale daje infeasible solution
- bardzo rzadko daje error bo solver proboje wartosci ktore sa infeasible
- Chyba problem ale im wieksza awersja do ryzyka tym wiecej bankructw

The talk will introduce a financial network contagion model, in which banks, by optimizing their balance sheet are endogenously forming interbank market. The banks differ with respect to their risk aversion and thus may be more or less active on interbank market or lend to more risky counterparties. 


The talk will introduce a financial network contagion model, in which heterogenous banks, by optimizing their balance sheet, are endogenously forming a network of credit links. The interbank market, converging to the equilibrium through tatonnement process, allows to facilitate the exchange of funds to more profitable investments from banks with worse investment opportunities. The banks differ with respect to their risk aversion when optimizing balance sheet and matching counterparties, which makes them more or less active on interbank market or lend to more risky counterparties. An exogenous  introduction of a credit shock into the model allows to investigate whether a system with the same risk aversion among banks on average but different higher order moments is more prone to default contagions and how . e.g. with a single extremely risk loving entity, analogical to the "super-spreaders" from epidemiology.