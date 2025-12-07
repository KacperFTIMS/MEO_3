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

        # Lista do przechowywania stanów macierzy alokacji i bazy
        self.step_by_step_data = []

    def log(self, message):
        self.logs.append(message)

    def display_table_in_logs(self, pretty_table_data, title):
        """Pomocnicza funkcja generująca prostą tabelę w logach dla celów śledzenia kroków startowych."""

        # Odtworzenie podstawowego formatowania tabularycznego dla logów operacyjnych
        self.log(f"\n--- {title} ---")
        s = pretty_table_data
        if s:
            # Użycie stałej szerokości dla czytelności (np. 8 znaków + margines)
            lens = [max(map(len, col)) for col in zip(*s)]
            # Ograniczenie szerokości do max 10 znaków, aby uniknąć nadmiernej szerokości
            max_len = 10
            lens = [min(l, max_len) for l in lens]

            fmt = ' | '.join('{{:^{}}}'.format(x) for x in lens)
            table = [fmt.format(*row) for row in s]

            # Dodanie linii separującej nagłówki
            sep_line = "-+-".join(["-" * l for l in lens])

            self.log(sep_line)
            # Wypisanie tabeli (bez ostatniego wiersza z sumą, który jest niepotrzebny w tym widoku)
            self.log('\n'.join(table[:-1]))
            self.log(sep_line)
            self.log(f"{fmt.format(*s[-1])}")

        self.log("-" * 30)

    def log_matrix(self, title):
        """
        Pomocnicza funkcja do zapisywania stanu macierzy alokacji i kosztów
        oraz wywoływania logowania do logów operacyjnych.
        """

        # ZAPIS STANU DO self.step_by_step_data
        if self.allocation is not None and self.basic_vars is not None:
            u, v = self.calculate_uv_without_degeneracy_fix()

            # Tworzenie czytelnej reprezentacji tabeli dla logów (na potrzeby podsumowania)
            temp_alloc = np.hstack((self.allocation, self.supply.reshape(-1, 1)))
            temp_alloc = np.vstack((temp_alloc, np.append(self.demand, np.sum(self.supply))))

            pretty_table_data = []
            for r_idx, row in enumerate(temp_alloc):
                r_str = []
                for c_idx, x in enumerate(row):
                    is_sum_row_col = (r_idx == self.rows or c_idx == self.cols)

                    if r_idx == self.rows and c_idx == self.cols:
                        r_str.append(f'={x:.0f}')
                    elif is_sum_row_col:
                        r_str.append(f'{x:.0f}')
                    elif abs(x) < self.epsilon:
                        r_str.append('0*' if self.basic_vars[r_idx, c_idx] else '0')
                    elif x.is_integer():
                        r_str.append(str(int(x)) if abs(x) > self.epsilon else '0')
                    else:
                        r_str.append(f'{x:.2f}')
                pretty_table_data.append(r_str)

            # Zapis stanu do step_by_step_data
            self.step_by_step_data.append({
                'title': title,
                'allocation': self.allocation.copy(),
                'basic_vars': self.basic_vars.copy(),
                'cost': self.get_total_cost(),
                'rows': self.rows,
                'cols': self.cols,
                'supply': self.supply.copy(),
                'demand': self.demand.copy(),
                'cost_matrix': self.cost_matrix.copy(),
                'pretty_table_data': pretty_table_data,  # Dane dla tabeli podsumowania
                'u': u,
                'v': v
            })

            # ZMIANA PRZYWRÓCONA: Wyświetlanie tabel w logach dla kroków startowych
            if 'NW:' in title or 'Min. Macierzy:' in title or 'Min. Wiersza:' in title or 'Min. Kolumny:' in title:
                self.display_table_in_logs(pretty_table_data, title)
            else:
                # Logowanie dla kroków MODI jest zwięzłe, bo pełna tabela jest poniżej
                self.log(f"\n--- {title} ---")
                self.log(f"    Całkowity koszt/zysk: {self.get_total_cost():.2f}")
                self.log("-" * 30)

    def prepare_data(self):
        self.cost_matrix = self.original_cost_matrix.copy()
        self.supply = self.original_supply.copy()
        self.demand = self.original_demand.copy()

        if self.type == 'max':
            self.log("Tryb Maksymalizacji: Negacja macierzy kosztów/zysków.")
            self.cost_matrix = -self.cost_matrix

        temp_supply = self.supply[self.supply != float('inf')]
        temp_demand = self.demand[self.demand != float('inf')]

        total_supply = np.sum(temp_supply)
        total_demand = np.sum(temp_demand)

        if abs(total_supply - total_demand) > self.epsilon:
            if total_supply > total_demand:
                diff = total_supply - total_demand
                self.log(
                    f"Bilansowanie: Podaż ({total_supply:.2f}) > Popyt ({total_demand:.2f}). Dodano wirtualnego odbiorcę: {diff:.2f}")
                dummy_col = np.zeros((self.cost_matrix.shape[0], 1))
                self.cost_matrix = np.hstack((self.cost_matrix, dummy_col))
                self.demand = np.append(self.demand, diff)

            else:
                diff = total_demand - total_supply
                self.log(
                    f"Bilansowanie: Popyt ({total_demand:.2f}) > Podaż ({total_supply:.2f}). Dodano wirtualnego dostawcę: {diff:.2f}")
                dummy_row = np.zeros((1, self.cost_matrix.shape[1]))
                self.cost_matrix = np.vstack((self.cost_matrix, dummy_row))
                self.supply = np.append(self.supply, diff)
        else:
            self.log(f"Bilansowanie: Zadanie jest zbilansowane (Suma: {total_supply:.2f}).")

        self.rows = len(self.supply)
        self.cols = len(self.demand)
        self.allocation = np.zeros((self.rows, self.cols))
        self.basic_vars = np.zeros((self.rows, self.cols), dtype=bool)

    # --- METODY STARTOWE (Z logowaniem kroków) ---

    def nw_corner_method(self):
        """Metoda Kąta Północno-Zachodniego (NW)"""
        self.log("\nMetoda kąta północno - zachodniego")
        supply = self.supply.copy()
        demand = self.demand.copy()
        i, j = 0, 0
        step = 1
        while i < self.rows and j < self.cols:
            quantity = min(supply[i], demand[j])
            self.allocation[i, j] = quantity
            self.basic_vars[i, j] = True

            self.log(f"Krok NW {step}: Przydzielono {quantity} -> Komórka [{i}, {j}]")
            self.log_matrix(f"NW Krok {step}")

            supply[i] -= quantity
            demand[j] -= quantity

            if abs(supply[i]) < self.epsilon and abs(demand[j]) < self.epsilon:
                if i + 1 < self.rows:
                    self.basic_vars[i + 1, j] = True
                    i += 1
                elif j + 1 < self.cols:
                    self.basic_vars[i, j + 1] = True
                    j += 1
                else:
                    i += 1
            elif abs(supply[i]) < self.epsilon:
                i += 1
            else:
                j += 1
            step += 1

        self.log(f"Koszt/Zysk rozwiązania początkowego (NW): {self.get_total_cost():.2f}")

    def matrix_min_method(self):
        """Metoda Minimalnego Elementu Macierzy"""
        self.log("\n Metoda minimalnego elementu macierzy")
        supply = self.supply.copy()
        demand = self.demand.copy()
        cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                cells.append((self.cost_matrix[r, c], r, c))
        cells.sort(key=lambda x: x[0])

        step = 1
        for cost, r, c in cells:
            if supply[r] > 0 and demand[c] > 0:
                quantity = min(supply[r], demand[c])
                self.allocation[r, c] = quantity
                self.basic_vars[r, c] = True
                self.log(f"Krok Min. Macierzy {step}: Koszt {cost:.2f}: Przydzielono {quantity} -> Komórka [{r}, {c}]")
                self.log_matrix(f"Min. Macierzy: Alokacja w kroku {step}")

                supply[r] -= quantity
                demand[c] -= quantity
                step += 1

        self.log(f"Koszt/Zysk rozwiązania początkowego (Min. Macierzy): {self.get_total_cost():.2f}")

    def row_min_method(self):
        """Metoda Minimalnego Elementu w Wierszu"""
        self.log("\nMetoda mnimalnego elementu w wierszu")
        supply = self.supply.copy()
        demand = self.demand.copy()
        step = 1

        for r in range(self.rows):
            self.log(f"Analiza wiersza {r} (Dostawca {r})...")

            current_supply = supply.copy()
            current_demand = demand.copy()

            while current_supply[r] > self.epsilon:
                min_cost = float('inf')
                target_c = -1
                for c in range(self.cols):
                    if current_demand[c] > self.epsilon and self.cost_matrix[r, c] < min_cost:
                        min_cost = self.cost_matrix[r, c]
                        target_c = c

                if target_c == -1: break

                quantity = min(current_supply[r], current_demand[target_c])

                self.allocation[r, target_c] += quantity
                self.basic_vars[r, target_c] = True

                self.log(
                    f"Krok Min. Wiersza {step}: Min w wierszu to koszt {min_cost:.2f}: Przydzielono {quantity} -> [{r}, {target_c}]")
                self.log_matrix(f"Min. Wiersza: Alokacja w kroku {step}")

                current_supply[r] -= quantity
                current_demand[target_c] -= quantity
                supply[r] -= quantity
                demand[target_c] -= quantity
                step += 1

        self.log(f"Koszt/Zysk rozwiązania początkowego (Min. Wiersza): {self.get_total_cost():.2f}")

    def col_min_method(self):
        """Metoda Minimalnego Elementu w Kolumnie"""
        self.log("\nMetoda minimalnego elementu w kolumnie")
        supply = self.supply.copy()
        demand = self.demand.copy()
        step = 1

        for c in range(self.cols):
            self.log(f"Analiza kolumny {c} (Odbiorca {c})...")

            current_supply = supply.copy()
            current_demand = demand.copy()

            while current_demand[c] > self.epsilon:
                min_cost = float('inf')
                target_r = -1
                for r in range(self.rows):
                    if current_supply[r] > self.epsilon and self.cost_matrix[r, c] < min_cost:
                        min_cost = self.cost_matrix[r, c]
                        target_r = r

                if target_r == -1: break

                quantity = min(current_supply[target_r], current_demand[c])

                self.allocation[target_r, c] += quantity
                self.basic_vars[target_r, c] = True

                self.log(
                    f"Krok Min. Kolumny {step}: Min w kolumnie to koszt {min_cost:.2f}: Przydzielono {quantity} -> [{target_r}, {c}]")
                self.log_matrix(f"Min. Kolumny: Alokacja w kroku {step}")

                current_supply[target_r] -= quantity
                current_demand[c] -= quantity
                supply[target_r] -= quantity
                demand[c] -= quantity
                step += 1

        self.log(f"Koszt/Zysk rozwiązania początkowego (Min. Kolumny): {self.get_total_cost():.2f}")

    # --- METODA POTENCJAŁÓW ---

    def solve_potentials(self):
        """Główna pętla metody potencjałów"""
        self.log("\n==========================================")
        self.log(" Metoda potencjałów")
        self.log("==========================================")

        iteration = 0
        initial_cost = self.get_total_cost()
        max_iterations = self.rows * self.cols * 2

        while iteration < max_iterations:
            iteration += 1
            self.log(f"\n--- Iteracja {iteration} (Koszt: {initial_cost:.2f}) ---")

            self.handle_degeneracy()

            u, v = self.calculate_uv_without_degeneracy_fix()

            if None in u or None in v:
                self.log("Błąd: Niestabilna baza")
                break

            deltas = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if not self.basic_vars[r, c]:
                        delta = self.cost_matrix[r, c] - (u[r] + v[c])
                        deltas.append((delta, r, c))

            # Warunek optymalności: Wszystkie delty >= 0
            if not deltas or min(d[0] for d in deltas) >= -self.epsilon:
                self.log("\nWszystkie delty >= 0. Rozwiązanie jest optymalne")
                self.log_matrix("Rozwiązanie jest optymalne")
                break

            # Wybór zmiennej wchodzącej: Najbardziej ujemna delta
            entering = min(deltas, key=lambda x: x[0])

            # Dopasowanie komunikatu do trybu
            if self.type == 'min':
                self.log(
                    rf"Rozwiązanie nieoptymalne. Najbardziej ujemna delta {entering[0]:.2f}")
            else:  # type == 'max'
                # Delta dla pierwotnego zysku to -Delta_kosztu
                original_gain_delta = -entering[0]
                self.log(
                    rf"Rozwiązanie nieoptymalne. Największy wzrost zysku na jednostkę: {original_gain_delta:+.2f}")

            start_node = (entering[1], entering[2])
            self.log(f"Zmienna wchodząca do bazy: Wiersz {start_node[0]}, Kolumna {start_node[1]}")

            path = self.find_cycle(start_node)

            if not path:
                self.log("Błąd: Nie znaleziono poprawnego cyklu")
                break

            path_str = " -> ".join([f"[{r},{c}]" for r, c in path])
            self.log(f"Znaleziono cykl: [{start_node[0]},{start_node[1]}] (+) -> {path_str}")

            minus_cells_values = []
            for i in range(len(path)):
                if (i + 1) % 2 != 0:
                    r, c = path[i]
                    minus_cells_values.append((self.allocation[r, c], r, c))

            if not minus_cells_values: break

            theta_val, r_leave, c_leave = min(minus_cells_values, key=lambda x: x[0])
            theta = max(0.0, theta_val)

            if theta < self.epsilon:
                theta = 0
                self.log(f"Wartość przesunięcia = 0.00. Zmiana zmiennej bazowej")
            else:
                self.log(f"Wartość przesunięcia = {theta:.2f}")

            self.allocation[start_node] += theta
            self.basic_vars[start_node] = True

            leaving_vars_candidates = []

            for i, (r, c) in enumerate(path):
                sign = (-1) ** (i + 1)
                self.allocation[r, c] += sign * theta

                if abs(self.allocation[r, c]) < self.epsilon and self.basic_vars[r, c]:
                    self.allocation[r, c] = 0
                    leaving_vars_candidates.append((r, c))

            if not leaving_vars_candidates: break

            r_out, c_out = min(leaving_vars_candidates, key=lambda x: (x[0], x[1]))
            self.basic_vars[r_out, c_out] = False
            self.log(f"Zmienna opuszczająca bazę:  [{r_out},{c_out}]")

            new_cost = self.get_total_cost()
            self.log(f"Łączny koszt/zysk po iteracji: {new_cost:.2f}")

            if self.type == 'min' and new_cost > initial_cost + self.epsilon:
                self.log(f"Błąd koszt się zwiększył: {initial_cost:.2f} -> {new_cost:.2f}")
                break
            elif self.type == 'max' and new_cost < initial_cost - self.epsilon:
                self.log(f"Błąd zysk zmalał: {initial_cost:.2f} -> {new_cost:.2f}")
                break

            initial_cost = new_cost

            self.log_matrix(f"Tabela po iteracji {iteration}")

    def handle_degeneracy(self):
        num_basic = np.sum(self.basic_vars)
        required = self.rows + self.cols - 1

        if num_basic > required:
            self.log(f"Nadmiar bazowych: {num_basic} > {required}")
            removed_count = 0

            for r in range(self.rows):
                for c in range(self.cols):
                    if self.basic_vars[r, c] and abs(self.allocation[r, c]) < self.epsilon:
                        self.basic_vars[r, c] = False
                        u, v = self.calculate_uv_without_degeneracy_fix()

                        if None not in u and None not in v:
                            self.log(f"    Usunięto zbędne zero bazowe z [{r},{c}].")
                            removed_count += 1
                        else:
                            self.basic_vars[r, c] = True

                        if np.sum(self.basic_vars) == required: break
                if np.sum(self.basic_vars) == required: break

            if removed_count > 0: self.log(f"    Usunięto {removed_count} zbędnych zer bazowych.")

        num_basic = np.sum(self.basic_vars)
        if num_basic < required:
            diff = required - num_basic
            self.log(f"Liczba zmiennych bazowych {num_basic} < {required}. Dodaję zera bazowe.")
            added = 0

            candidates = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if not self.basic_vars[r, c] and abs(self.allocation[r, c]) < self.epsilon:
                        candidates.append((self.cost_matrix[r, c], r, c))

            candidates.sort(key=lambda x: x[0])

            for cost, r, c in candidates:
                if added >= diff: break
                self.basic_vars[r, c] = True
                u, v = self.calculate_uv_without_degeneracy_fix()

                if None in u or None in v:
                    self.basic_vars[r, c] = False
                else:
                    self.allocation[r, c] = 0
                    self.log(f"    Dodano zero bazowe (0*) do pola [{r},{c}] (koszt {cost:.2f}).")
                    added += 1

    def calculate_uv_without_degeneracy_fix(self):
        u = [None] * self.rows
        v = [None] * self.cols
        if self.rows > 0:
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

    def find_cycle(self, start_pos):
        r_start, c_start = start_pos
        self.basic_vars[r_start, c_start] = True

        def dfs_cycle_search(current_pos, path, mode):
            r, c = current_pos

            neighbors = []
            if mode == 'row':
                for j in range(self.cols):
                    if j != c and self.basic_vars[r, j]:
                        neighbors.append((r, j))
            else:
                for i in range(self.rows):
                    if i != r and self.basic_vars[i, c]:
                        neighbors.append((i, c))

            next_mode = 'col' if mode == 'row' else 'row'

            for n in neighbors:
                if n == start_pos and len(path) >= 3:
                    return True

                if n not in path:
                    path.append(n)
                    if dfs_cycle_search(n, path, next_mode):
                        return True
                    path.pop()

            return False

        cycle = []
        if dfs_cycle_search(start_pos, cycle, 'row'):
            self.basic_vars[r_start, c_start] = False
            return cycle

        cycle = []
        if dfs_cycle_search(start_pos, cycle, 'col'):
            self.basic_vars[r_start, c_start] = False
            return cycle

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
            return -total  # zwracamy pierwotny zysk
        return total


# --- INTERFEJS GRAFICZNY (APP) ---
class TransportApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozwiązywanie zadania transportowego")
        self.root.geometry("1000x800")

        # --- MENU GÓRNE (POMOC) ---
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Pomoc", menu=help_menu)
        help_menu.add_command(label="Instrukcja", command=self.show_help_message)

        # --- Panel Sterowania ---
        control_frame = ttk.LabelFrame(root, text="Ustawienia")
        control_frame.pack(fill="x", padx=10, pady=5)

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

        ttk.Label(control_frame, text="Cel:").grid(row=0, column=2, padx=5, pady=5)
        self.opt_type = tk.StringVar(value="min")
        ttk.Radiobutton(control_frame, text="Minimalizacja Kosztów", variable=self.opt_type, value="min").grid(row=0,
                                                                                                               column=3)
        ttk.Radiobutton(control_frame, text="Maksymalizacja Zysku", variable=self.opt_type, value="max").grid(row=0,
                                                                                                              column=4)

        ttk.Label(control_frame, text="Dostawcy:").grid(row=1, column=0)
        self.rows_entry = ttk.Entry(control_frame, width=5)
        self.rows_entry.insert(0, "3")
        self.rows_entry.grid(row=1, column=1)

        ttk.Label(control_frame, text="Odbiorcy:").grid(row=1, column=2)
        self.cols_entry = ttk.Entry(control_frame, width=5)
        self.cols_entry.insert(0, "4")
        self.cols_entry.grid(row=1, column=3)

        ttk.Button(control_frame, text="Generuj Tabelę", command=self.generate_table).grid(row=1, column=5, padx=10)

        ttk.Label(control_frame, text="Przykłady:").grid(row=2, column=0)
        self.preset_combo = ttk.Combobox(control_frame, values=[
            "Zadanie 1 (3x4 Min)",
            "Zadanie 2 (4x4 Max - Przydział)",
            "Zadanie 4 (3x4 Min)",
            "Zadanie 5 (3x3 Min - Niezbilansowane)",
            "Zadanie 6 (3x4 Min - Koszty z odl.)",
            "Zadanie 7 (3x4 Min - Niezbilansowane)",
            "Zadanie 9 (4x3 Min)",
            "Zadanie 10 (4x3 Max)",
            "Zadanie 13 (3x4 Min - Czas)",
            "Zadanie 14a (3x3 Min)",
            "Zadanie 14b (3x3 Min - Blokada)",
        ], state="readonly", width=35)
        self.preset_combo.grid(row=2, column=1, columnspan=2)
        ttk.Button(control_frame, text="Załaduj Przykład", command=self.load_preset).grid(row=2, column=3)

        # --- Panel Tabeli Danych Wejściowych ---
        self.table_frame = ttk.LabelFrame(root, text="Dane wejściowe")
        self.table_frame.pack(fill="x", padx=10, pady=5)

        # --- Panel Wyników i Logów ---
        bottom_frame = ttk.Frame(root)
        bottom_frame.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Button(bottom_frame, text="Rozwiąż", command=self.solve).pack(pady=5)

        self.log_text = tk.Text(bottom_frame, height=35, wrap=tk.NONE)
        self.log_text.pack(fill="both", expand=True)

        # Dodanie suwaka poziomego do pola logów
        self.h_scrollbar = ttk.Scrollbar(self.log_text, orient=tk.HORIZONTAL, command=self.log_text.xview)
        self.log_text.config(xscrollcommand=self.h_scrollbar.set)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.cells = []
        self.supply_entries = []
        self.demand_entries = []

        self.generate_table()

    def load_preset(self):
        selection = self.preset_combo.get()
        if not selection: return

        costs = []
        supply = []
        demand = []

        if "Zadanie 1 " in selection:
            costs = [[10, 40, 50, 20], [20, 60, 40, 60], [30, 30, 30, 40]]
            supply = [300, 450, 800]
            demand = [630, 160, 170, 340]
            self.opt_type.set("min")
        elif "Zadanie 2" in selection:
            costs = [[30, 50, 60, 80], [40, 80, 70, 100], [60, 40, 50, 30], [90, 60, 60, 70]]
            supply = [1, 1, 1, 1]
            demand = [1, 1, 1, 1]
            self.opt_type.set("max")
        elif "Zadanie 4" in selection or "Zadanie 8" in selection:
            costs = [[3, 4, 7, 1], [5, 1, 3, 2], [2, 4, 5, 4]]
            supply = [100, 150, 100]
            demand = [80, 120, 120, 30]
            self.opt_type.set("min")
        elif "Zadanie 5" in selection:
            costs = [[3, 7, 4], [4, 9, 6]]
            supply = [100, 200]
            demand = [80, 150, 170]
            self.opt_type.set("min")
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
        elif "Zadanie 7" in selection:
            costs = [[10, 40, 50, 20], [20, 60, 40, 60], [30, 30, 30, 40]]
            supply = [400, 600, 550]
            demand = [500, 350, 300, 700]
            self.opt_type.set("min")
        elif "Zadanie 9" in selection:
            costs = [[9, 5, 3], [7, 8, 2], [2, 10, 5], [4, 6, 7]]
            supply = [20, 30, 25, 40]
            demand = [16, 34, 50]
            self.opt_type.set("min")
        elif "Zadanie 10" in selection:
            costs = [[1, 3, 4], [5, 8, 6], [1, 2, 5], [2, 1, 7]]
            supply = [600, 500, 300, 400]
            demand = [300, 900, 600]
            self.opt_type.set("max")
        elif "Zadanie 13" in selection:
            costs = [[1, 3, 7, 2], [2, 2, 2, 3], [1, 3, 6, 5]]
            supply = [100, 200, 150]
            demand = [80, 170, 90, 110]
            self.opt_type.set("min")
        elif "Zadanie 14a" in selection:
            costs = [[40, 80, 60], [30, 60, 50], [90, 40, 30]]
            supply = [70, 30, 100]
            demand = [50, 60, 90]
            self.opt_type.set("min")
        elif "Zadanie 14b" in selection:
            costs = [[40, 80, 60], [30, 60, 50], [90, 40, 30]]
            costs[1][0] = float('inf')
            supply = [70, 30, 100]
            demand = [50, 60, 90]
            self.opt_type.set("min")
        else:
            return

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
                    val = str(costs[i][j])
                    if val.lower() == "inf":
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

        ttk.Label(self.table_frame, text="Dost\\Odb").grid(row=0, column=0)
        for j in range(cols):
            ttk.Label(self.table_frame, text=f"Odb {j + 1}").grid(row=0, column=j + 1)
        ttk.Label(self.table_frame, text="Podaż").grid(row=0, column=cols + 1)

        for i in range(rows):
            ttk.Label(self.table_frame, text=f"Dost {i + 1}").grid(row=i + 1, column=0)
            row_cells = []
            for j in range(cols):
                e = ttk.Entry(self.table_frame, width=8)
                e.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                e.insert(0, "0")
                row_cells.append(e)
            self.cells.append(row_cells)

            s = ttk.Entry(self.table_frame, width=8)
            s.grid(row=i + 1, column=cols + 1, padx=5)
            s.insert(0, "0")
            self.supply_entries.append(s)

        ttk.Label(self.table_frame, text="Popyt").grid(row=rows + 1, column=0)
        for j in range(cols):
            d = ttk.Entry(self.table_frame, width=8)
            d.grid(row=rows + 1, column=j + 1, padx=1, pady=5)
            d.insert(0, "0")
            self.demand_entries.append(d)

    def format_step_table(self, data, step_index):
        """
        Generuje JEDNOLINIOWY, sformatowany tekst tabeli dla danego kroku,
        używany GŁÓWNIE dla iteracji MODI.
        """
        rows = data['rows']
        cols = data['cols']
        u = data['u']
        v = data['v']

        # Sprawdzenie, czy to jest krok startowy (który nie ma jeszcze obliczonych U/V
        # ani delty)
        is_initial_step = (None in u or None in v)

        # Zwiększona szerokość kolumny, aby pomieścić C, X i Delta w jednej linii
        CELL_WIDTH = 25
        COL_HEADER_WIDTH = 10
        SIDE_WIDTH = 12

        output = [
            f"\n\n=========================================================================================",
            f"Krok {step_index + 1}: {data['title']} (Koszt/Zysk: {data['cost']:.2f})",
            f"========================================================================================="
        ]

        # 1. Nagłówki kolumn
        header_parts = [f"{'Dost\\Odb':<{SIDE_WIDTH}}"]
        for j in range(cols):
            header_parts.append(f"| {'Odb ' + str(j + 1):^{CELL_WIDTH}}")
        header_parts.append(f"| {'Podaż':^{COL_HEADER_WIDTH}}")
        header_parts.append(f"| {'U':^{SIDE_WIDTH}} |")
        output.append("".join(header_parts))
        output.append("-" * (SIDE_WIDTH + (CELL_WIDTH + 2) * cols + (COL_HEADER_WIDTH + 2) + SIDE_WIDTH + 3))

        # 2. Wiersze danych (Alokacja + Koszt/Delta)
        for i in range(rows):
            row_parts = [f"{'Dost ' + str(i + 1):<{SIDE_WIDTH}}"]
            for j in range(cols):
                alloc_val = data['pretty_table_data'][i][j]
                is_basic = data['basic_vars'][i, j]
                cost_val = data['cost_matrix'][i, j]

                # Formatowanie C (Koszt) / Zysk (dla trybu max)
                cost_text = ""
                if self.opt_type.get() == 'max':
                    original_cost_val = -cost_val
                    # Wyświetl zysk
                    if abs(original_cost_val) > 10000:
                        cost_text = f"Zysk=INF" if original_cost_val > 0 else "Zysk=-INF"
                    else:
                        cost_text = f"Zysk={original_cost_val:.0f}"
                else:
                    # Wyświetl koszt
                    cost_text = f"C={cost_val:.0f}" if cost_val != float('inf') and abs(
                        cost_val) < 10000 else "C=INF"

                # Formatowanie Delta (tylko dla kroków MODI)
                delta_text = ""
                if not is_initial_step and not is_basic:
                    delta = cost_val - (u[i] + v[j])

                    if self.opt_type.get() == 'max':
                        # W trybie 'max' pokazujemy różnicę pierwotnego zysku: Delta_ZYSKU = -Delta_KOSZTU
                        delta_gain = -delta
                        delta_text = f" \u0394Z={delta_gain:+.2f}"
                    else:
                        delta_text = f" \u0394C={delta:+.2f}"  # znak + dla lepszej czytelności

                # Łączenie w jedną linię: [X=...* C=... Delta=...]
                basic_mark = '*' if is_basic else ' '
                cell_content = f"X:{alloc_val}{basic_mark} | {cost_text}{delta_text}"

                row_parts.append(f"| {cell_content:<{CELL_WIDTH}}")

            # Podaż i Potencjał U
            supply_val = data['pretty_table_data'][i][cols]
            u_val_raw = u[i]
            u_val = f"{u_val_raw:+.2f}" if u_val_raw is not None else "---"

            row_parts.append(f"| {supply_val:>{COL_HEADER_WIDTH}}")
            row_parts.append(f"| {u_val:^{SIDE_WIDTH}} |")

            output.append("".join(row_parts))
            output.append("-" * (SIDE_WIDTH + (CELL_WIDTH + 2) * cols + (COL_HEADER_WIDTH + 2) + SIDE_WIDTH + 3))

        # 3. Wiersz Popytu
        demand_row_parts = [f"{'Popyt':<{SIDE_WIDTH}}"]
        for j in range(cols):
            demand_val = data['pretty_table_data'][rows][j]
            demand_row_parts.append(f"| {demand_val:^{CELL_WIDTH}}")

        demand_row_parts.append(f"| {'':^{COL_HEADER_WIDTH}}")
        demand_row_parts.append(f"| {'':^{SIDE_WIDTH}} |")
        output.append("".join(demand_row_parts))
        output.append("-" * (SIDE_WIDTH + (CELL_WIDTH + 2) * cols + (COL_HEADER_WIDTH + 2) + SIDE_WIDTH + 3))

        # 4. Wiersz Potencjałów V
        v_row_parts = [f"{'V':<{SIDE_WIDTH}}"]
        for j in range(cols):
            v_val_raw = v[j]
            v_val = f"{v_val_raw:+.2f}" if v_val_raw is not None else "---"
            v_row_parts.append(f"| {v_val:^{CELL_WIDTH}}")

        v_row_parts.append(f"| {'':^{COL_HEADER_WIDTH}}")
        v_row_parts.append(f"| {'':^{SIDE_WIDTH}} |")
        output.append("".join(v_row_parts))
        output.append("\n")

        return "\n".join(output)

    def solve(self):
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

        solver = TransportSolver(cost_matrix, supply, demand, type=self.opt_type.get())
        solver.prepare_data()

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

        solver.solve_potentials()

        self.log_text.delete(1.0, tk.END)

        # 1. Wypisz Logi operacyjne (W TYM KLASYCZNE TABELE POCZĄTKOWE)

        # Lista logów do pominięcia (dotyczy zwięzłego logowania MODI, które nie jest tabelą)
        skipped_logs = []
        for log in solver.logs:
            # W logach pozostawiamy wszystko, w tym tabele startowe (z display_table_in_logs)
            self.log_text.insert(tk.END, log + "\n")
            # Dodatkowo, zapisujemy logi z MODI, które powielają się poniżej (np. --- Iteracja 1 ---)
            if "Iteracja" in log or "Rozwiązanie jest optymalne" in log:
                skipped_logs.append(log)

        self.log_text.insert(tk.END, "\n\n" + "=" * 50 + "\n")
        self.log_text.insert(tk.END, "--- Rozwiązanie krok po kroku ---\n")
        self.log_text.insert(tk.END, "=" * 50 + "\n")

        # 2. Wypisz Tabela Krok po Kroku (TYLKO KROKI MODI)
        modi_step_index = 0
        for i, step_data in enumerate(solver.step_by_step_data):
            # Sprawdzamy, czy u i v zostały obliczone (oznacza krok MODI)
            if None not in step_data['u'] and None not in step_data['v']:
                # Jeśli to jest krok MODI, wyświetlamy go w formacie jednoliniiowym
                formatted_table = self.format_step_table(step_data, modi_step_index)
                self.log_text.insert(tk.END, formatted_table + "\n\n")
                modi_step_index += 1

        # 3. Pokaż wynik końcowy
        total = solver.get_total_cost()
        opt_type_text = "zysk" if self.opt_type.get() == 'max' else "koszt"
        self.log_text.insert(tk.END, f"\n\nOptymalny {opt_type_text}: {total:.2f}")

    def show_help_message(self):
        msg = (
            "INSTRUKCJA KORZYSTANIA Z PROGRAMU:\n\n"
            "1. Wybierz metodę startową (np. Kąt  - zachodni).\n"
            "   Służy ona do znalezienia pierwszego rozwiązania.\n\n"
            "2. Wybierz cel: Minimalizacja kosztów (standard) lub\n"
            "   Maksymalizacja zysku (dla zadań z zyskiem).\n\n"
            "3. Wpisz liczbę dostawców i odbiorców, a następnie\n"
            "   kliknij 'Generuj Tabelę'.\n\n"
            "4. Uzupełnij tabelę danymi:\n"
            "   - Środek: Jednostkowe koszty transportu.\n"
            "   - Prawa kolumna: Podaż (ile towaru ma dostawca).\n"
            "   - Dolny wiersz: Popyt (ile towaru potrzebuje odbiorca).\n\n"
            "5. Możesz też wybrać gotowy przykład z listy 'Przykłady'\n"
            "   i kliknąć 'Załaduj Przykład'.\n\n"
            "6. Kliknij 'Rozwiąż'. Program najpierw wyznaczy rozwiązanie\n"
            "   startowe wybraną metodą, a następnie zoptymalizuje je\n"
            "   Metodą Potencjałów aż do wyniku optymalnego."
        )
        messagebox.showinfo("Pomoc", msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = TransportApp(root)
    root.mainloop()