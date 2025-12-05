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

    def log(self, message):
        self.logs.append(message)

    def log_matrix(self, title):
        """Pomocnicza funkcja do ładnego wypisywania macierzy w logach"""
        self.log(f"\n--- {title} ---")
        if self.allocation is not None:
            # Formatowanie tabeli
            s = [[str(int(x)) if x.is_integer() else str(x) for x in row] for row in self.allocation]
            lens = [max(map(len, col)) for col in zip(*s)]
            fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
            table = [fmt.format(*row) for row in s]
            self.log('\n'.join(table))
        self.log("-" * 30)

    def prepare_data(self):
        self.cost_matrix = self.original_cost_matrix.copy()
        self.supply = self.original_supply.copy()
        self.demand = self.original_demand.copy()

        if self.type == 'max':
            self.log("Tryb Maksymalizacji: Negacja macierzy kosztów/zysków.")
            self.cost_matrix = -self.cost_matrix

        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)

        # FIX: Dodano epsilon, aby uniknąć błędów zaokrągleń (np. 200.0 vs 200.0000001)
        epsilon = 1e-9

        if total_supply > total_demand + epsilon:
            diff = total_supply - total_demand
            self.log(
                f"BILANSOWANIE: Podaż ({total_supply}) > Popyt ({total_demand}). Dodano fikcyjnego odbiorcę: {diff}")
            dummy_col = np.zeros((self.cost_matrix.shape[0], 1))
            self.cost_matrix = np.hstack((self.cost_matrix, dummy_col))
            self.demand = np.append(self.demand, diff)

        elif total_demand > total_supply + epsilon:
            diff = total_demand - total_supply
            self.log(
                f"BILANSOWANIE: Popyt ({total_demand}) > Podaż ({total_supply}). Dodano fikcyjnego dostawcę: {diff}")
            dummy_row = np.zeros((1, self.cost_matrix.shape[1]))
            self.cost_matrix = np.vstack((self.cost_matrix, dummy_row))
            self.supply = np.append(self.supply, diff)
        else:
            self.log(f"BILANSOWANIE: Zadanie jest zbilansowane (Suma: {total_supply}).")

        self.rows = len(self.supply)
        self.cols = len(self.demand)
        self.allocation = np.zeros((self.rows, self.cols))
        self.basic_vars = np.zeros((self.rows, self.cols), dtype=bool)

    # --- METODY STARTOWE ---

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

            if supply[i] == 0 and demand[j] == 0:
                # Obsługa jednoczesnego wyczerpania (degeneracja)
                if i + 1 < self.rows:
                    i += 1
                else:
                    j += 1
            elif supply[i] == 0:
                i += 1
            else:
                j += 1
        self.log_matrix("Rozwiązanie początkowe (NW)")
        self.log(f"Koszt/Zysk rozwiązania początkowego: {self.get_total_cost()}")

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
                self.log(f"Koszt {cost}: Przydzielono {quantity} -> Komórka [{r}, {c}]")
                supply[r] -= quantity
                demand[c] -= quantity
        self.log_matrix("Rozwiązanie początkowe (Min. Macierzy)")
        self.log(f"Koszt/Zysk rozwiązania początkowego: {self.get_total_cost()}")

    def row_min_method(self):
        """Metoda Minimalnego Elementu w Wierszu"""
        self.log("\n>>> START: Metoda Minimalnego Elementu w Wierszu")
        supply = self.supply.copy()
        demand = self.demand.copy()

        for r in range(self.rows):
            self.log(f"Analiza wiersza {r} (Dostawca {r})...")
            while supply[r] > 0:
                min_cost = float('inf')
                target_c = -1
                for c in range(self.cols):
                    if demand[c] > 0 and self.cost_matrix[r, c] < min_cost:
                        min_cost = self.cost_matrix[r, c]
                        target_c = c

                if target_c == -1: break

                quantity = min(supply[r], demand[target_c])
                self.allocation[r, target_c] = quantity
                self.basic_vars[r, target_c] = True
                self.log(f"  Min w wierszu to koszt {min_cost}: Przydzielono {quantity} -> [{r}, {target_c}]")
                supply[r] -= quantity
                demand[target_c] -= quantity
        self.log_matrix("Rozwiązanie początkowe (Min. Wiersza)")
        self.log(f"Koszt/Zysk rozwiązania początkowego: {self.get_total_cost()}")

    def col_min_method(self):
        """Metoda Minimalnego Elementu w Kolumnie"""
        self.log("\n>>> START: Metoda Minimalnego Elementu w Kolumnie")
        supply = self.supply.copy()
        demand = self.demand.copy()

        for c in range(self.cols):
            self.log(f"Analiza kolumny {c} (Odbiorca {c})...")
            while demand[c] > 0:
                min_cost = float('inf')
                target_r = -1
                for r in range(self.rows):
                    if supply[r] > 0 and self.cost_matrix[r, c] < min_cost:
                        min_cost = self.cost_matrix[r, c]
                        target_r = r

                if target_r == -1: break

                quantity = min(supply[target_r], demand[c])
                self.allocation[target_r, c] = quantity
                self.basic_vars[target_r, c] = True
                self.log(f"  Min w kolumnie to koszt {min_cost}: Przydzielono {quantity} -> [{target_r}, {c}]")
                supply[target_r] -= quantity
                demand[c] -= quantity
        self.log_matrix("Rozwiązanie początkowe (Min. Kolumny)")
        self.log(f"Koszt/Zysk rozwiązania początkowego: {self.get_total_cost()}")

    # --- METODA POTENCJAŁÓW ---

    def solve_potentials(self):
        """Główna pętla metody potencjałów"""
        self.log("\n==========================================")
        self.log(" ROZPOCZYNAM OPTYMALIZACJĘ METODĄ POTENCJAŁÓW")
        self.log("==========================================")

        iteration = 0
        while True:
            iteration += 1
            self.log(f"\n--- ITERACJA {iteration} ---")

            self.handle_degeneracy()
            u, v = self.calculate_uv()
            if u is None:
                self.log("Błąd obliczania potencjałów.")
                break

            # Wypisz potencjały (rzutowanie na float dla czytelności)
            self.log(f"Potencjały u (wiersze): {[round(float(x), 2) for x in u]}")
            self.log(f"Potencjały v (kolumny): {[round(float(x), 2) for x in v]}")

            deltas = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if not self.basic_vars[r, c]:
                        delta = self.cost_matrix[r, c] - (u[r] + v[c])
                        deltas.append((delta, r, c))

            deltas.sort(key=lambda x: x[0])

            if not deltas or deltas[0][0] >= -1e-9:
                self.log("\n>>> KONIEC: Wszystkie delty >= 0. Rozwiązanie jest optymalne!")
                break

            entering = deltas[0]
            self.log(f"Rozwiązanie nieoptymalne. Najbardziej ujemna delta: {entering[0]:.2f}")
            self.log(f"Zmienna wchodząca do bazy: Wiersz {entering[1]}, Kolumna {entering[2]}")

            start_node = (entering[1], entering[2])
            path = self.find_cycle(start_node)

            if not path:
                self.log("Błąd: Nie znaleziono cyklu.")
                break

            path_str = " -> ".join([f"[{r},{c}]" for r, c in path]) + f" -> [{start_node[0]},{start_node[1]}]"
            self.log(f"Znaleziono cykl zmian: {path_str}")

            minus_cells_values = []
            for i in range(1, len(path), 2):
                r, c = path[i]
                minus_cells_values.append(self.allocation[r, c])

            theta = min(minus_cells_values)
            self.log(f"Wartość przesunięcia (theta) = {theta} (minimum z pól oznaczonych '-')")

            for i, (r, c) in enumerate(path):
                if i == 0:
                    self.allocation[r, c] += theta
                    self.basic_vars[r, c] = True
                elif i % 2 == 1:
                    self.allocation[r, c] -= theta
                    if self.allocation[r, c] == 0 and self.basic_vars[r, c]:
                        if theta == minus_cells_values[i // 2]:
                            self.basic_vars[r, c] = False
                            theta = -1
                else:
                    self.allocation[r, c] += theta

            self.log_matrix(f"Tabela po iteracji {iteration}")

    def handle_degeneracy(self):
        num_basic = np.sum(self.basic_vars)
        required = self.rows + self.cols - 1
        if num_basic < required:
            diff = required - num_basic
            self.log(f"[!] DEGENERACJA: Liczba zmiennych bazowych {num_basic} < {required}.")
            added = 0
            while added < diff:
                min_c = float('inf')
                candidate = None
                for r in range(self.rows):
                    for c in range(self.cols):
                        if not self.basic_vars[r, c]:
                            if self.cost_matrix[r, c] < min_c:
                                min_c = self.cost_matrix[r, c]
                                candidate = (r, c)
                if candidate:
                    self.allocation[candidate] = 0
                    self.basic_vars[candidate] = True
                    self.log(f"    Dodano epsilon (0) do pola {candidate}, aby umożliwić obliczenia.")
                    added += 1
                else:
                    break

    def calculate_uv(self):
        u = [None] * self.rows
        v = [None] * self.cols
        u[0] = 0
        changed = True
        while changed:
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
        if None in u or None in v:
            u = [0 if x is None else x for x in u]
            v = [0 if x is None else x for x in v]
        return u, v

    def find_cycle(self, start_pos):
        def get_possible_moves(curr_pos, mode):
            moves = []
            r, c = curr_pos
            if mode == 'horizontal':
                for j in range(self.cols):
                    if j != c and (self.basic_vars[r, j] or (r, j) == start_pos):
                        moves.append((r, j))
            else:
                for i in range(self.rows):
                    if i != r and (self.basic_vars[i, c] or (i, c) == start_pos):
                        moves.append((i, c))
            return moves

        visited = []
        path = []

        def dfs(current, mode):
            path.append(current)
            if current == start_pos and len(path) > 1: return True
            next_mode = 'vertical' if mode == 'horizontal' else 'horizontal'
            neighbors = get_possible_moves(current, mode)
            for n in neighbors:
                if n == start_pos and len(path) >= 3: return True
                if n not in path:
                    if dfs(n, next_mode): return True
            path.pop()
            return False

        if dfs(start_pos, 'horizontal'):
            path.append(start_pos)
            return path[:-1]
        path = []
        if dfs(start_pos, 'vertical'):
            path.append(start_pos)
            return path[:-1]
        return None

    def get_total_cost(self):
        total = 0
        for r in range(self.rows):
            for c in range(self.cols):
                alloc = self.allocation[r, c]
                cost = self.cost_matrix[r, c]
                if alloc > 0:
                    if cost == float('inf'):
                        return float('inf')
                    if cost == float('-inf'):
                        return float('-inf')
                    total += alloc * cost
        if self.type == 'max': return -total
        return total


# --- INTERFEJS GRAFICZNY ---
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
        self.method_var = tk.StringVar(value="nw")
        methods = [
            ("Kąt Północno-Zachodni", "nw"),
            ("Min. Element Macierzy", "matrix_min"),
            ("Min. w Wierszu", "row_min"),
            ("Min. w Kolumnie", "col_min")
        ]
        self.method_combo = ttk.Combobox(control_frame, values=[m[0] for m in methods], state="readonly")
        self.method_combo.current(0)
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

    def load_preset(self):
        selection = self.preset_combo.get()
        if not selection: return

        costs = []
        supply = []
        demand = []

        # [cite_start]--- ZADANIE 1 [cite: 1428-1432] ---
        # Zakłady odzieżowe. Podaż > Popyt (Niezbilansowane)
        if "Zadanie 1 " in selection:
            costs = [[10, 40, 50, 20], [20, 60, 40, 60], [30, 30, 30, 40]]
            supply = [300, 450, 800]
            demand = [630, 160, 170, 340]
            self.opt_type.set("min")

        # [cite_start]--- ZADANIE 2 [cite: 1435-1439] ---
        # Przemysł gumowy. Przydział (1 maszyna = 1 wyrób). Maksymalizacja produkcji.
        elif "Zadanie 2" in selection:
            costs = [[30, 50, 60, 80], [40, 80, 70, 100], [60, 40, 50, 30], [90, 60, 60, 70]]
            supply = [1, 1, 1, 1]
            demand = [1, 1, 1, 1]
            self.opt_type.set("max")

        # [cite_start]--- ZADANIE 4 [cite: 8-12] ---
        # Hurtownie.
        elif "Zadanie 4" in selection or "Zadanie 8" in selection:
            costs = [[3, 4, 7, 1], [5, 1, 3, 2], [2, 4, 5, 4]]
            supply = [100, 150, 100]
            demand = [80, 120, 120, 30]
            self.opt_type.set("min")

        # [cite_start]--- ZADANIE 5 [cite: 1416-1420] ---
        # Bawełna. Egipt/Rosja. Popyt > Podaż (Niedobór).
        elif "Zadanie 5" in selection:
            costs = [[3, 7, 4], [4, 9, 6]]
            supply = [100, 200]
            demand = [80, 150, 170]
            self.opt_type.set("min")

        # [cite_start]--- ZADANIE 6 [cite: 1-6] ---
        # Półfabrykaty. Koszt = Odległość * Współczynnik.
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

        # [cite_start]--- ZADANIE 7 [cite: 20-23] ---
        # Tartaki. Popyt > Podaż.
        elif "Zadanie 7" in selection:
            costs = [[10, 40, 50, 20], [20, 60, 40, 60], [30, 30, 30, 40]]
            supply = [400, 600, 550]
            demand = [500, 350, 300, 700]
            self.opt_type.set("min")

        # [cite_start]--- ZADANIE 9 [cite: 1400-1406] ---
        # Krosna. Minimalizacja nakładów.
        elif "Zadanie 9" in selection:
            costs = [[9, 5, 3], [7, 8, 2], [2, 10, 5], [4, 6, 7]]
            supply = [20, 30, 25, 40]
            demand = [16, 34, 50]
            self.opt_type.set("min")

        # [cite_start]--- ZADANIE 10 [cite: 1440-1444] ---
        # Jabłka. Maksymalizacja zysku.
        elif "Zadanie 10" in selection:
            costs = [[1, 3, 4], [5, 8, 6], [1, 2, 5], [2, 1, 7]]
            supply = [600, 500, 300, 400]
            demand = [300, 900, 600]
            self.opt_type.set("max")

        # [cite_start]--- ZADANIE 13 [cite: 1423-1426] ---
        # Mleczarnie. Minimalizacja czasu (traktowany jako koszt).
        elif "Zadanie 13" in selection:
            costs = [[1, 3, 7, 2], [2, 2, 2, 3], [1, 3, 6, 5]]
            supply = [100, 200, 150]
            demand = [80, 170, 90, 110]
            self.opt_type.set("min")

        # [cite_start]--- ZADANIE 14a [cite: 15-18] ---
        # Warzywa. Standardowe.
        elif "Zadanie 14a" in selection:
            costs = [[40, 80, 60], [30, 60, 50], [90, 40, 30]]
            supply = [70, 30, 100]
            demand = [50, 60, 90]
            self.opt_type.set("min")

        # Warzywa. Blokada trasy (Punkt 2 -> Przetwórnia 1).
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
            self.supply_entries[i].delete(0, tk.END)
            self.supply_entries[i].insert(0, str(supply[i]))
            for j in range(len(costs[0])):
                self.cells[i][j].delete(0, tk.END)
                self.cells[i][j].insert(0, str(costs[i][j]))

        for j in range(len(demand)):
            self.demand_entries[j].delete(0, tk.END)
            self.demand_entries[j].insert(0, str(demand[j]))

    def solve(self):
        # Pobierz dane
        try:
            rows = int(self.rows_entry.get())
            cols = int(self.cols_entry.get())
            cost_matrix = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    row.append(float(self.cells[i][j].get()))
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
        self.log_text.insert(tk.END, f"\n\nŁĄCZNY KOSZT/ZYSK: {total}")

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
            "   - Środek: Jednostkowe koszty transportu.\n"
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