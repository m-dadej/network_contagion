- sproboj dodac stope zwrotu z cashu - moze wtedy banki beda braly wiecej cashu i bedzie mniej bankructw wraz z wyzsza sigma.

Analiza jaki jest wynik w zaleznosci od charakterystyk pierwszego bankructwa? (e.g. czy jak maly bank padnie to moze byc duzy problem dla systemu?)


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
- Chyba problem ale im wieksza awersja do ryzyka tym wiecej bankructw
