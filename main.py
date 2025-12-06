import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


# --- KLASA LOGIKI (SOLVER) ---
class TransportSolver:
    def __init__(self, cost_matrix, supply, demand, type='min'):
        """
        Inicjalizacja solvera.
        cost_matrix: macierz kosztów (lista list lub numpy array)
        supply: wektor podaży (dostawcy)
        demand: wektor popytu (odbiorcy)
        type: 'min' dla minimalizacji kosztów, 'max' dla maksymalizacji zysku
        """
        self.original_cost_matrix = np.array(cost_matrix, dtype=float)
        self.original_supply = np.array(supply, dtype=float)
        self.original_demand = np.array(demand, dtype=float)
        self.type = type

        self.cost_matrix = None
        self.supply = None
        self.demand = None
        self.allocation = None
        self.logs = []
        self.rows = 0
        self.cols = 0
        self.basic_vars = None
        self.epsilon = 1e-9  # Globalna tolerancja dla zer

    def log(self, message):
        self.logs.append(message)

    def log_matrix(self, title):
        """Pomocnicza funkcja do ładnego wypisywania macierzy w logach"""
        self.log(f"\n--- {title} ---")
        if self.allocation is not None:

            s = []
            # Dodatkowa kolumna dla podaży
            temp_alloc = np.hstack((self.allocation, self.supply.reshape(-1, 1)))
            # Dodatkowy wiersz dla popytu
            temp_alloc = np.vstack((temp_alloc, np.append(self.demand, np.sum(self.supply))))

            for r_idx, row in enumerate(temp_alloc):
                r_str = []
                for c_idx, x in enumerate(row):
                    if r_idx == self.rows and c_idx == self.cols:  # Prawy dolny róg (suma)
                        r_str.append(f'={x:.0f}')
                    elif r_idx == self.rows or c_idx == self.cols:  # Podaż/Popyt
                        r_str.append(f'{x:.0f}')
                    elif abs(x) < self.epsilon:
                        # Wypisujemy '0*' dla zmiennej bazowej zerowej
                        if self.basic_vars[r_idx, c_idx]:
                            r_str.append('0*')
                        else:
                            r_str.append('0')
                    elif x.is_integer():
                        r_str.append(str(int(x)))
                    else:
                        r_str.append(f'{x:.2f}')
                s.append(r_str)

            # Wyrównanie do prawej
            if s:
                lens = [max(map(len, col)) for col in zip(*s)]
                fmt = '\t'.join('{{:>{}}}'.format(x) for x in lens)
                table = [fmt.format(*row) for row in s]
                self.log('\n'.join(table[:-1]))  # Bez sumy
                self.log(fmt.format(*s[-1]))  # Sama suma
        self.log("-" * 30)

    def prepare_data(self):
        self.cost_matrix = self.original_cost_matrix.copy()
        self.supply = self.original_supply.copy()
        self.demand = self.original_demand.copy()

        if self.type == 'max':
            self.log("Tryb Maksymalizacji: Negacja macierzy kosztów/zysków.")
            # W Metodzie Potencjałów, aby dążyć do maksimum,
            # zamieniamy problem na min. koszty ujemne (max. zyski)
            self.cost_matrix = -self.cost_matrix

        # Wyczyść nieskończoności w S/D przed bilansowaniem
        temp_supply = self.supply[self.supply != float('inf')]
        temp_demand = self.demand[self.demand != float('inf')]

        total_supply = np.sum(temp_supply)
        total_demand = np.sum(temp_demand)

        if abs(total_supply - total_demand) > self.epsilon:
            if total_supply > total_demand:
                diff = total_supply - total_demand
                self.log(
                    f"BILANSOWANIE: Podaż ({total_supply:.2f}) > Popyt ({total_demand:.2f}). Dodano fikcyjnego odbiorcę: {diff:.2f}")
                # Fikcyjny odbiorca - koszty 0
                dummy_col = np.zeros((self.cost_matrix.shape[0], 1))
                self.cost_matrix = np.hstack((self.cost_matrix, dummy_col))
                self.demand = np.append(self.demand, diff)

            else:  # total_demand > total_supply
                diff = total_demand - total_supply
                self.log(
                    f"BILANSOWANIE: Popyt ({total_demand:.2f}) > Podaż ({total_supply:.2f}). Dodano fikcyjnego dostawcę: {diff:.2f}")
                # Fikcyjny dostawca - koszty 0
                dummy_row = np.zeros((1, self.cost_matrix.shape[1]))
                self.cost_matrix = np.vstack((self.cost_matrix, dummy_row))
                self.supply = np.append(self.supply, diff)
        else:
            self.log(f"BILANSOWANIE: Zadanie jest zbilansowane (Suma: {total_supply:.2f}).")

        self.rows = len(self.supply)
        self.cols = len(self.demand)
        self.allocation = np.zeros((self.rows, self.cols))
        self.basic_vars = np.zeros((self.rows, self.cols), dtype=bool)

    # --- METODY STARTOWE (bez zmian) ---

    def nw_corner_method(self):
        """Metoda Kąta Północno-Zachodniego"""
        self.log("\n>>> START: Metoda Kąta Północno-Zachodniego")
        supply = self.supply.copy()
        demand = self.demand.copy()
        i, j = 0, 0
        while i < self.rows and j < self.cols:
            quantity = min(supply[i], demand[j])
            self.allocation[i, j] = quantity
            self.basic_vars[i, j] = True

            self.log(f"Przydzielono {quantity} -> Komórka [{i}, {j}]")

            supply[i] -= quantity
            demand[j] -= quantity

            if abs(supply[i]) < self.epsilon and abs(demand[j]) < self.epsilon:
                # W przypadku zerowego cyklu (dwie wartości jednocześnie się wyzerowały)
                # Musimy dodać jedną ze zmiennych do bazy jako '0*'
                if i + 1 < self.rows:
                    self.basic_vars[i + 1, j] = True
                    i += 1
                elif j + 1 < self.cols:
                    self.basic_vars[i, j + 1] = True
                    j += 1
                else:  # Ostatnie pole
                    i += 1  # Wychodzimy z pętli
            elif abs(supply[i]) < self.epsilon:
                i += 1
            else:
                j += 1
        self.log_matrix("Rozwiązanie początkowe (NW)")
        self.log(f"Koszt/Zysk rozwiązania początkowego: {self.get_total_cost():.2f}")

    def matrix_min_method(self):
        """Metoda Minimalnego Elementu Macierzy"""
        self.log("\n>>> START: Metoda Minimalnego Elementu Macierzy")
        supply = self.supply.copy()
        demand = self.demand.copy()
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                cells.append((self.cost_matrix[r, c], r, c))
        cells.sort(key=lambda x: x[0])

        for cost, r, c in cells:
            if supply[r] > 0 and demand[c] > 0:
                quantity = min(supply[r], demand[c])
                self.allocation[r, c] = quantity
                self.basic_vars[r, c] = True
                self.log(f"Koszt {cost:.2f}: Przydzielono {quantity} -> Komórka [{r}, {c}]")
                supply[r] -= quantity
                demand[c] -= quantity
        self.log_matrix("Rozwiązanie początkowe (Min. Macierzy)")
        self.log(f"Koszt/Zysk rozwiązania początkowego: {self.get_total_cost():.2f}")

    def row_min_method(self):
        """Metoda Minimalnego Elementu w Wierszu"""
        self.log("\n>>> START: Metoda Minimalnego Elementu w Wierszu")
        supply = self.supply.copy()
        demand = self.demand.copy()

        for r in range(self.rows):
            self.log(f"Analiza wiersza {r} (Dostawca {r})...")
            while supply[r] > self.epsilon:
                min_cost = float('inf')
                target_c = -1
                for c in range(self.cols):
                    if demand[c] > self.epsilon and self.cost_matrix[r, c] < min_cost:
                        min_cost = self.cost_matrix[r, c]
                        target_c = c

                if target_c == -1: break

                quantity = min(supply[r], demand[target_c])
                self.allocation[r, target_c] = quantity
                self.basic_vars[r, target_c] = True
                self.log(f"  Min w wierszu to koszt {min_cost:.2f}: Przydzielono {quantity} -> [{r}, {target_c}]")
                supply[r] -= quantity
                demand[target_c] -= quantity
        self.log_matrix("Rozwiązanie początkowe (Min. Wiersza)")
        self.log(f"Koszt/Zysk rozwiązania początkowego: {self.get_total_cost():.2f}")

    def col_min_method(self):
        """Metoda Minimalnego Elementu w Kolumnie"""
        self.log("\n>>> START: Metoda Minimalnego Elementu w Kolumnie")
        supply = self.supply.copy()
        demand = self.demand.copy()

        for c in range(self.cols):
            self.log(f"Analiza kolumny {c} (Odbiorca {c})...")
            while demand[c] > self.epsilon:
                min_cost = float('inf')
                target_r = -1
                for r in range(self.rows):
                    if supply[r] > self.epsilon and self.cost_matrix[r, c] < min_cost:
                        min_cost = self.cost_matrix[r, c]
                        target_r = r

                if target_r == -1: break

                quantity = min(supply[target_r], demand[c])
                self.allocation[target_r, c] = quantity
                self.basic_vars[target_r, c] = True
                self.log(f"  Min w kolumnie to koszt {min_cost:.2f}: Przydzielono {quantity} -> [{target_r}, {c}]")
                supply[target_r] -= quantity
                demand[c] -= quantity
        self.log_matrix("Rozwiązanie początkowe (Min. Kolumny)")
        self.log(f"Koszt/Zysk rozwiązania początkowego: {self.get_total_cost():.2f}")

    # --- METODA POTENCJAŁÓW (POPRAWIONA LOGIKA CYKLU) ---

    def solve_potentials(self):
        """Główna pętla metody potencjałów"""
        self.log("\n==========================================")
        self.log(" ROZPOCZYNAM OPTYMALIZACJĘ METODĄ POTENCJAŁÓW")
        self.log("==========================================")

        iteration = 0
        initial_cost = self.get_total_cost()
        max_iterations = self.rows * self.cols * 2  # Ograniczenie liczby iteracji

        while iteration < max_iterations:
            iteration += 1
            self.log(f"\n--- ITERACJA {iteration} (Koszt: {initial_cost:.2f}) ---")

            # Krok 1: Obsługa degeneracji (jeśli liczba zmiennych bazowych jest zbyt mała)
            self.handle_degeneracy()

            # Krok 2: Obliczanie potencjałów (u i v)
            u, v = self.calculate_uv()

            if None in u or None in v:
                self.log("Błąd krytyczny: Nie można obliczyć wszystkich potencjałów. Prawdopodobna niestabilność bazy.")
                break

            # Krok 3: Obliczanie macierzy delt (kosztów względnych)
            deltas = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if not self.basic_vars[r, c]:
                        delta = self.cost_matrix[r, c] - (u[r] + v[c])
                        deltas.append((delta, r, c))

            # Kryterium optymalności (Min: delta >= 0 ; Max: delta <= 0)
            if self.type == 'min':
                # Szukamy najmniejszej ujemnej delty (Delta < -epsilon)
                if not deltas or min(d[0] for d in deltas) >= -self.epsilon:
                    self.log("\n>>> KONIEC: Wszystkie delty >= 0. Rozwiązanie jest optymalne!")
                    break
                # Najbardziej ujemna delta (zmienna wchodząca)
                entering = min(deltas, key=lambda x: x[0])
                self.log(f"Rozwiązanie nieoptymalne. Najbardziej ujemna delta: {entering[0]:.2f}")
            else:  # self.type == 'max' (Pracujemy na macierzy -C, więc szukamy max ujemnej, czyli min C)
                # Szukamy największej dodatniej delty (Delta > epsilon)
                # Macierz kosztów jest zanegowana, więc delta = -cost_original - (u+v)
                # Musimy szukać największej dodatniej delty, co odpowiada najmniejszemu ujemnemu kosztowi względnemu w oryginalnym problemie.
                if not deltas or max(d[0] for d in deltas) <= self.epsilon:
                    self.log(
                        "\n>>> KONIEC: Wszystkie delty <= 0 (w problemie min. negatywnych kosztów). Rozwiązanie jest optymalne!")
                    break
                # Najbardziej dodatnia delta
                entering = max(deltas, key=lambda x: x[0])
                self.log(
                    f"Rozwiązanie nieoptymalne. Najbardziej dodatnia delta: {entering[0]:.2f} (Największy potencjał poprawy zysku).")

            start_node = (entering[1], entering[2])
            self.log(f"Zmienna wchodząca do bazy: Wiersz {start_node[0]}, Kolumna {start_node[1]}")

            # Krok 4: Znalezienie cyklu
            path = self.find_cycle(start_node)

            if not path:
                self.log("Błąd: Nie znaleziono poprawnego cyklu zamkniętego. Optymalizacja zatrzymana.")
                break

            path_str = " -> ".join([f"[{r},{c}]" for r, c in path])
            self.log(f"Znaleziono cykl: [{start_node[0]},{start_node[1]}] (+) -> {path_str}")

            # Krok 5: Wyznaczanie theta (minimum z pól ujemnych)
            minus_cells_values = []

            # W cyklu: start_node (+), path[0] (-), path[1] (+), path[2] (-), ...
            # Pola ujemne to path[0], path[2], path[4], ... (indeksy parzyste w path)
            for i in range(len(path)):
                if (i + 1) % 2 != 0:  # Indeksy 0, 2, 4, ...
                    r, c = path[i]
                    minus_cells_values.append((self.allocation[r, c], r, c))

            if not minus_cells_values:
                self.log("Błąd: Błąd cyklu (brak pól ujemnych do wyboru theta).")
                break

            # Theta to najmniejsza alokacja w polach ujemnych
            theta = min(m[0] for m in minus_cells_values)

            if theta < self.epsilon:
                theta = 0
                self.log(f"Wartość przesunięcia (theta) = 0.00. Wymiana zmiennej bazowej (degeneracja).")
            else:
                self.log(f"Wartość przesunięcia (theta) = {theta:.2f}")

            # --- Krok 6: Przejście do nowego rozwiązania bazowego ---

            # 1. Zmienna wchodząca do bazy
            self.allocation[start_node] += theta
            self.basic_vars[start_node] = True

            leaving_vars_candidates = []

            # 2. Aktualizacja zmiennych wzdłuż cyklu (path)
            for i, (r, c) in enumerate(path):
                # path[i] odpowiada pozycji, która powinna mieć znak (-1)^(i+1)
                sign = (-1) ** (i + 1)  # i=0 -> (-1), i=1 -> (+1), i=2 -> (-1)

                self.allocation[r, c] += sign * theta

                # Kandydaci na zmienne opuszczające bazę (alokacja bliska zera)
                if abs(self.allocation[r, c]) < self.epsilon and self.basic_vars[r, c]:
                    self.allocation[r, c] = 0  # Wymuszenie zera
                    leaving_vars_candidates.append((r, c))

            # 3. Kryterium wyjścia: Usuwamy tylko jedną zmienną z bazy
            if not leaving_vars_candidates:
                self.log("Błąd: Żadna zmienna nie opuściła bazy (brak zera w półcyklu ujemnym).")
                break

            # W przypadku degeneracji (theta=0 lub więcej niż jeden kandydat)
            # Wybieramy tego kandydata, który jest w zbiorze minus_cells_values, a następnie wybieramy jeden z nich.

            # Wystarczy wybrać dowolną zmienną z listy kandydatów
            r, c = leaving_vars_candidates[0]
            self.basic_vars[r, c] = False
            self.log(f"Zmienna opuszczająca bazę: [{r},{c}]")

            # Weryfikacja kosztu (musi maleć lub być stały)
            new_cost = self.get_total_cost()
            self.log(f"Łączny koszt/zysk po iteracji: {new_cost:.2f}")

            # Sprawdzanie, czy koszt nie wzrósł (dla min)
            if self.type == 'min' and new_cost > initial_cost + self.epsilon:
                self.log(f"!!! KRYTYCZNY BŁĄD !!! KOSZT WZRÓSŁ: {initial_cost:.2f} -> {new_cost:.2f}")
                break
            # Sprawdzanie, czy zysk nie zmalał (dla max)
            elif self.type == 'max' and new_cost < initial_cost - self.epsilon:
                self.log(f"!!! KRYTYCZNY BŁĄD !!! ZYSK ZMALAŁ: {initial_cost:.2f} -> {new_cost:.2f}")
                break

            initial_cost = new_cost

            self.log_matrix(f"Tabela po iteracji {iteration}")

    def handle_degeneracy(self):
        # Sprawdzanie i usuwanie zer bazowych, które nie są niezbędne

        num_basic = np.sum(self.basic_vars)
        required = self.rows + self.cols - 1

        if num_basic > required:
            # Nadmierna liczba zmiennych bazowych - usuwamy zbędne '0*'
            self.log(f"[!] Nadmiar bazowych: {num_basic} > {required}. Usuwam zbędne '0*'.")
            removed_count = 0

            # Szukamy zmiennych bazowych z alokacją 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.basic_vars[r, c] and abs(self.allocation[r, c]) < self.epsilon:
                        # Sprawdzamy, czy usunięcie tej zmiennej nie spowoduje problemu

                        # Tymczasowo usuwamy z bazy
                        self.basic_vars[r, c] = False

                        # Upewniamy się, że pozostałe bazowe nadal pozwalają na obliczenie u/v
                        u, v = self.calculate_uv_without_degeneracy_fix()

                        if None not in u and None not in v:
                            # Nadal da się obliczyć u/v, więc ta zmienna była zbędna
                            self.log(f"    Usunięto zbędne zero bazowe z [{r},{c}].")
                            removed_count += 1
                        else:
                            # Ta zmienna bazowa jest niezbędna do obliczenia u/v
                            self.basic_vars[r, c] = True

                        if np.sum(self.basic_vars) == required:
                            break
                if np.sum(self.basic_vars) == required:
                    break

            if removed_count > 0:
                self.log(f"    Usunięto {removed_count} zbędnych zer bazowych.")

        # Dodawanie zer bazowych (jeśli liczba zmiennych bazowych jest zbyt mała)
        num_basic = np.sum(self.basic_vars)
        if num_basic < required:
            diff = required - num_basic
            self.log(f"[!] DEGENERACJA: Liczba zmiennych bazowych {num_basic} < {required}. Dodaję zera bazowe.")
            added = 0

            # Wyszukujemy najlepszych kandydatów (najmniejszy koszt)
            candidates = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if not self.basic_vars[r, c] and abs(self.allocation[r, c]) < self.epsilon:
                        candidates.append((self.cost_matrix[r, c], r, c))

            candidates.sort(key=lambda x: x[0])

            for cost, r, c in candidates:
                if added >= diff: break

                # Dodajemy tymczasowo jako bazowe
                self.basic_vars[r, c] = True

                # Sprawdzamy, czy dodanie nie tworzy cyklu (czyli czy u/v jest rozwiązywalne)
                u, v = self.calculate_uv_without_degeneracy_fix()

                if None in u or None in v:
                    # Dodanie stworzyło cykl w bazie - cofamy
                    self.basic_vars[r, c] = False
                else:
                    # Jest to dobre pole do dodania
                    self.allocation[r, c] = 0
                    self.log(f"    Dodano zero bazowe (0*) do pola [{r},{c}] (koszt {cost:.2f}).")
                    added += 1

            if added < diff:
                self.log(
                    "!!! Ostrzeżenie !!! Nie udało się dodać wystarczającej liczby zer bazowych do stworzenia niecyklicznej bazy.")

    def calculate_uv_without_degeneracy_fix(self):
        # Funkcja pomocnicza do sprawdzania, czy można obliczyć u/v
        u = [None] * self.rows
        v = [None] * self.cols
        u[0] = 0.0
        changed = True

        while changed and (None in u or None in v):
            changed = False
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.basic_vars[r, c]:
                        if u[r] is not None and v[c] is None:
                            v[c] = self.cost_matrix[r, c] - u[r]
                            changed = True
                        elif u[r] is None and v[c] is not None:
                            u[r] = self.cost_matrix[r, c] - v[c]
                            changed = True
        return u, v

    def calculate_uv(self):
        # Główna funkcja obliczająca u/v
        u, v = self.calculate_uv_without_degeneracy_fix()

        # Opcjonalnie: Ustawienie pozostałych None na 0.0 w przypadku cyklu zerowego w bazie
        # W normalnym przypadku, jeśli handle_degeneracy działa, nie powinno być None
        u = [0.0 if x is None else x for x in u]
        v = [0.0 if x is None else x for x in v]

        return u, v

    def find_cycle(self, start_pos):
        """
        [POPRAWIONA IMPLEMENTACJA]
        Znajduje zamkniętą ścieżkę (cykl) dla zmiennej wchodzącej start_pos
        używając tylko zmiennych bazowych. Wykorzystuje DFS.
        """
        r_start, c_start = start_pos
        # Tymczasowe włączenie zmiennej wchodzącej do bazy
        self.basic_vars[r_start, c_start] = True

        # path to lista krotek (r, c)
        def dfs_cycle_search(current_pos, path, mode):
            r, c = current_pos

            # Wyszukiwanie sąsiadów - tylko wzdłuż zmiennych bazowych
            neighbors = []
            if mode == 'row':  # Szukamy w kolumnie (ruch poziomy)
                for j in range(self.cols):
                    if j != c and self.basic_vars[r, j]:
                        neighbors.append((r, j))
            else:  # mode == 'col' # Szukamy w wierszu (ruch pionowy)
                for i in range(self.rows):
                    if i != r and self.basic_vars[i, c]:
                        neighbors.append((i, c))

            # Właściwy ruch: przełączanie z wiersza na kolumnę i na odwrót
            next_mode = 'col' if mode == 'row' else 'row'

            for n in neighbors:
                # Jeśli wróciliśmy do punktu startowego, znaleźliśmy cykl.
                # Cykl musi mieć co najmniej 4 wierzchołki: start -> A -> B -> start. (3 w path)
                if n == start_pos and len(path) >= 3:
                    return True

                # Jeśli sąsiad nie jest jeszcze w ścieżce
                if n not in path:
                    path.append(n)
                    if dfs_cycle_search(n, path, next_mode):
                        return True
                    path.pop()  # Backtrack

            return False

        cycle = []
        # Próba rozpoczęcia ruchu w poziomie
        if dfs_cycle_search(start_pos, cycle, 'row'):
            self.basic_vars[r_start, c_start] = False
            return cycle

        cycle = []
        # Próba rozpoczęcia ruchu w pionie
        if dfs_cycle_search(start_pos, cycle, 'col'):
            self.basic_vars[r_start, c_start] = False
            return cycle

        # Usuwamy tymczasowe włączenie, jeśli cykl nie został znaleziony
        self.basic_vars[r_start, c_start] = False
        return None

    def get_total_cost(self):
        total = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                alloc = self.allocation[r, c]
                cost = self.cost_matrix[r, c]
                if abs(alloc) > self.epsilon:
                    if cost == float('inf'):
                        return float('inf')
                    if cost == float('-inf'):
                        return float('-inf')
                    total += alloc * cost

        if self.type == 'max':
            # Jeśli tryb to 'max', macierz cost_matrix była zanegowana,
            # więc wynik jest ujemny. Zwracamy -total, aby dostać zysk.
            return -total
        return total


# --- INTERFEJS GRAFICZNY (bez zmian) ---
class TransportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozwiązywanie Zadania Transportowego i Przydziału")
        self.root.geometry("1000x750")

        # --- MENU GÓRNE (POMOC) ---
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Pomoc", menu=help_menu)
        help_menu.add_command(label="Instrukcja", command=self.show_help_message)

        # --- Panel Sterowania ---
        control_frame = ttk.LabelFrame(root, text="Ustawienia")
        control_frame.pack(fill="x", padx=10, pady=5)

        # Wybór metody startowej
        ttk.Label(control_frame, text="Metoda startowa:").grid(row=0, column=0, padx=5, pady=5)
        self.method_var = tk.StringVar(value="matrix_min")
        methods = [
            ("Kąt Północno-Zachodni", "nw"),
            ("Min. Element Macierzy", "matrix_min"),
            ("Min. w Wierszu", "row_min"),
            ("Min. w Kolumnie", "col_min")
        ]
        self.method_combo = ttk.Combobox(control_frame, values=[m[0] for m in methods], state="readonly")
        self.method_combo.current(1)
        self.method_combo.grid(row=0, column=1, padx=5, pady=5)
        self.method_map = {m[0]: m[1] for m in methods}

        # Typ optymalizacji
        ttk.Label(control_frame, text="Cel:").grid(row=0, column=2, padx=5, pady=5)
        self.opt_type = tk.StringVar(value="min")
        ttk.Radiobutton(control_frame, text="Minimalizacja Kosztów", variable=self.opt_type, value="min").grid(row=0,
                                                                                                               column=3)
        ttk.Radiobutton(control_frame, text="Maksymalizacja Zysku", variable=self.opt_type, value="max").grid(row=0,
                                                                                                              column=4)

        # Wymiary
        ttk.Label(control_frame, text="Dostawcy:").grid(row=1, column=0)
        self.rows_entry = ttk.Entry(control_frame, width=5)
        self.rows_entry.insert(0, "3")
        self.rows_entry.grid(row=1, column=1)

        ttk.Label(control_frame, text="Odbiorcy:").grid(row=1, column=2)
        self.cols_entry = ttk.Entry(control_frame, width=5)
        self.cols_entry.insert(0, "4")
        self.cols_entry.grid(row=1, column=3)

        ttk.Button(control_frame, text="Generuj Tabelę", command=self.generate_table).grid(row=1, column=5, padx=10)

        # Presety zadań
        ttk.Label(control_frame, text="Przykłady:").grid(row=2, column=0)
        self.preset_combo = ttk.Combobox(control_frame, values=[
            "Zadanie 1 (3x4 Min)",
            "Zadanie 2 (4x4 Max - Przydział)",
            "Zadanie 4 (3x4 Min)",
            "Zadanie 5 (3x3 Min - Niezbilansowane)",
            "Zadanie 6 (3x4 Min - Koszty z odl.)",
            "Zadanie 7 (3x4 Min - Niezbilansowane)",
            "Zadanie 8 (3x4 Min - to samo co Zad 4)",
            "Zadanie 9 (4x3 Min)",
            "Zadanie 10 (4x3 Max)",
            "Zadanie 13 (3x4 Min - Czas)",
            "Zadanie 14a (3x3 Min)",
            "Zadanie 14b (3x3 Min - Blokada)",
        ], state="readonly", width=35)
        self.preset_combo.grid(row=2, column=1, columnspan=2)
        ttk.Button(control_frame, text="Załaduj Przykład", command=self.load_preset).grid(row=2, column=3)

        # --- Panel Tabeli ---
        self.table_frame = ttk.Frame(root)
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # --- Panel Wyników ---
        bottom_frame = ttk.Frame(root)
        bottom_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Button(bottom_frame, text="ROZWIĄŻ (Metoda Potencjałów)", command=self.solve).pack(pady=5)

        self.log_text = tk.Text(bottom_frame, height=15)
        self.log_text.pack(fill="both", expand=True)

        self.cells = []
        self.supply_entries = []
        self.demand_entries = []

        # Startowa tabela
        self.generate_table()

    def load_preset(self):
        selection = self.preset_combo.get()
        if not selection: return

        costs = []
        supply = []
        demand = []

        # --- ZADANIE 1 ---
        if "Zadanie 1 " in selection:
            costs = [[10, 40, 50, 20], [20, 60, 40, 60], [30, 30, 30, 40]]
            supply = [300, 450, 800]
            demand = [630, 160, 170, 340]
            self.opt_type.set("min")

        # --- ZADANIE 2 (Wymyślone dla testów - przydział) ---
        elif "Zadanie 2" in selection:
            costs = [[30, 50, 60, 80], [40, 80, 70, 100], [60, 40, 50, 30], [90, 60, 60, 70]]
            supply = [1, 1, 1, 1]
            demand = [1, 1, 1, 1]
            self.opt_type.set("max")

        # --- ZADANIE 4/8 (Hurtownie - Wymyślone) ---
        elif "Zadanie 4" in selection or "Zadanie 8" in selection:
            costs = [[3, 4, 7, 1], [5, 1, 3, 2], [2, 4, 5, 4]]
            supply = [100, 150, 100]
            demand = [80, 120, 120, 30]
            self.opt_type.set("min")

        # --- ZADANIE 5 (Bawełna. Popyt > Podaż - Wymyślone) ---
        elif "Zadanie 5" in selection:
            costs = [[3, 7, 4], [4, 9, 6]]
            supply = [100, 200]
            demand = [80, 150, 170]
            self.opt_type.set("min")

        # --- ZADANIE 6 (Koszty z odl. - Wymyślone) ---
        elif "Zadanie 6" in selection:
            dist = [[76, 20, 30, 36], [20, 10, 40, 24], [18, 44, 10, 16]]
            coeffs = [0.5, 1.0, 2.0]
            costs = []
            for r in range(3):
                row_costs = [d * coeffs[r] for d in dist[r]]
                costs.append(row_costs)
            supply = [1200, 900, 900]
            demand = [600, 500, 800, 700]
            self.opt_type.set("min")

        # --- ZADANIE 7 (Tartaki. Popyt > Podaż - Wymyślone) ---
        elif "Zadanie 7" in selection:
            costs = [[10, 40, 50, 20], [20, 60, 40, 60], [30, 30, 30, 40]]
            supply = [400, 600, 550]
            demand = [500, 350, 300, 700]
            self.opt_type.set("min")

        # --- ZADANIE 9 (Krosna. Min. nakładów - Wymyślone) ---
        elif "Zadanie 9" in selection:
            costs = [[9, 5, 3], [7, 8, 2], [2, 10, 5], [4, 6, 7]]
            supply = [20, 30, 25, 40]
            demand = [16, 34, 50]
            self.opt_type.set("min")

        # --- ZADANIE 10 (Jabłka. Max. zysku - Wymyślone) ---
        elif "Zadanie 10" in selection:
            costs = [[1, 3, 4], [5, 8, 6], [1, 2, 5], [2, 1, 7]]
            supply = [600, 500, 300, 400]
            demand = [300, 900, 600]
            self.opt_type.set("max")

        # --- ZADANIE 13 (Mleczarnie. Min. czasu - Wymyślone) ---
        elif "Zadanie 13" in selection:
            costs = [[1, 3, 7, 2], [2, 2, 2, 3], [1, 3, 6, 5]]
            supply = [100, 200, 150]
            demand = [80, 170, 90, 110]
            self.opt_type.set("min")

        # --- ZADANIE 14a (Warzywa - Wymyślone) ---
        elif "Zadanie 14a" in selection:
            costs = [[40, 80, 60], [30, 60, 50], [90, 40, 30]]
            supply = [70, 30, 100]
            demand = [50, 60, 90]
            self.opt_type.set("min")

        # --- ZADANIE 14b (Warzywa. Blokada trasy - Wymyślone) ---
        elif "Zadanie 14b" in selection:
            costs = [[40, 80, 60], [30, 60, 50], [90, 40, 30]]
            costs[1][0] = float('inf')  # Blokada
            supply = [70, 30, 100]
            demand = [50, 60, 90]
            self.opt_type.set("min")

        else:
            return

        # Wypełnij GUI danymi
        self.rows_entry.delete(0, tk.END);
        self.rows_entry.insert(0, str(len(costs)))
        self.cols_entry.delete(0, tk.END);
        self.cols_entry.insert(0, str(len(costs[0])))
        self.generate_table()

        for i in range(len(costs)):
            if i < len(self.supply_entries):
                self.supply_entries[i].delete(0, tk.END)
                self.supply_entries[i].insert(0, str(supply[i]))
            for j in range(len(costs[0])):
                if i < len(self.cells) and j < len(self.cells[i]):
                    self.cells[i][j].delete(0, tk.END)
                    # Wstaw 'inf' dla nieskończonych kosztów
                    val = str(costs[i][j])
                    if val == "inf":
                        val = "inf"
                    self.cells[i][j].insert(0, val)

        for j in range(len(demand)):
            if j < len(self.demand_entries):
                self.demand_entries[j].delete(0, tk.END)
                self.demand_entries[j].insert(0, str(demand[j]))

    def generate_table(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
        except ValueError:
            return

        self.cells = []
        self.supply_entries = []
        self.demand_entries = []

        # Nagłówki
        ttk.Label(self.table_frame, text="Dost\\Odb").grid(row=0, column=0)
        for j in range(cols):
            ttk.Label(self.table_frame, text=f"Odb {j + 1}").grid(row=0, column=j + 1)
        ttk.Label(self.table_frame, text="Podaż").grid(row=0, column=cols + 1)

        # Macierz
        for i in range(rows):
            ttk.Label(self.table_frame, text=f"Dost {i + 1}").grid(row=i + 1, column=0)
            row_cells = []
            for j in range(cols):
                e = ttk.Entry(self.table_frame, width=8)
                e.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                e.insert(0, "0")
                row_cells.append(e)
            self.cells.append(row_cells)

            # Podaż
            s = ttk.Entry(self.table_frame, width=8)
            s.grid(row=i + 1, column=cols + 1, padx=5)
            s.insert(0, "0")
            self.supply_entries.append(s)

        # Popyt
        ttk.Label(self.table_frame, text="Popyt").grid(row=rows + 1, column=0)
        for j in range(cols):
            d = ttk.Entry(self.table_frame, width=8)
            d.grid(row=rows + 1, column=j + 1, padx=1, pady=5)
            d.insert(0, "0")
            self.demand_entries.append(d)

    def solve(self):
        # Pobierz dane
        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
            cost_matrix = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    val = self.cells[i][j].get().strip()
                    if val.lower() in ('inf', 'm'):
                        row.append(float('inf'))
                    else:
                        row.append(float(val))
                cost_matrix.append(row)

            supply = [float(e.get()) for e in self.supply_entries]
            demand = [float(e.get()) for e in self.demand_entries]
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne liczby.")
            return

        # Utwórz solver
        solver = TransportSolver(cost_matrix, supply, demand, type=self.opt_type.get())
        solver.prepare_data()

        # Wybierz metodę startową
        method_name = self.method_combo.get()
        method_key = self.method_map[method_name]

        if method_key == "nw":
            solver.nw_corner_method()
        elif method_key == "matrix_min":
            solver.matrix_min_method()
        elif method_key == "row_min":
            solver.row_min_method()
        elif method_key == "col_min":
            solver.col_min_method()

        # Rozwiąż
        solver.solve_potentials()

        # Wyświetl logi
        self.log_text.delete(1.0, tk.END)
        for log in solver.logs:
            self.log_text.insert(tk.END, log + "\n")

        # Pokaż wynik końcowy w logach
        self.log_text.insert(tk.END, "\n--- MACIERZ ALOKACJI KOŃCOWEJ ---\n")
        self.log_text.insert(tk.END, str(solver.allocation))
        total = solver.get_total_cost()
        self.log_text.insert(tk.END, f"\n\nŁĄCZNY KOSZT/ZYSK OPTYMALNY: {total:.2f}")

    def show_help_message(self):
        msg = (
            "INSTRUKCJA KORZYSTANIA Z PROGRAMU:\n\n"
            "1. Wybierz metodę startową (np. Kąt Północno-Zachodni).\n"
            "   Służy ona do znalezienia pierwszego rozwiązania.\n\n"
            "2. Wybierz cel: Minimalizacja kosztów (standard) lub\n"
            "   Maksymalizacja zysku (dla zadań z zyskiem).\n\n"
            "3. Wpisz liczbę dostawców i odbiorców, a następnie\n"
            "   kliknij 'Generuj Tabelę'.\n\n"
            "4. Uzupełnij tabelę danymi:\n"
            "   - Środek: Jednostkowe koszty transportu (możesz wpisać 'inf' lub 'M' dla blokady trasy).\n"
            "   - Prawa kolumna: Podaż (ile towaru ma dostawca).\n"
            "   - Dolny wiersz: Popyt (ile towaru potrzebuje odbiorca).\n\n"
            "5. Możesz też wybrać gotowy przykład z listy 'Przykłady'\n"
            "   i kliknąć 'Załaduj Przykład'.\n\n"
            "6. Kliknij 'ROZWIĄŻ'. Program najpierw wyznaczy rozwiązanie\n"
            "   startowe wybraną metodą, a następnie zoptymalizuje je\n"
            "   Metodą Potencjałów aż do wyniku idealnego."
        )
        messagebox.showinfo("Pomoc", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = TransportApp(root)
    root.mainloop()