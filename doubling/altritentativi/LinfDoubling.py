import math
import numpy as np
from functools import lru_cache
#ALGORITMO PER I CALCOLO ESATTO DELLA DD SENZA PYTORCH E BASATo SU INSIEMI  

def distanza_linf_matrix(X: np.ndarray) -> np.ndarray:
    """
    Calcola la matrice delle distanze L_infinito per punti in R^512.

    X: array shape (n, 512)
    ritorna: matrice D shape (n, n), con
             D[i, j] = ||X[i] - X[j]||_infinito
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if d != 512:
        raise ValueError("I punti devono essere in R^512, quindi X deve avere shape (n, 512).")

    D = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        # distanza L_infinito: max delle differenze assolute coordinata per coordinata
        D[i, :] = np.max(np.abs(X[i] - X), axis=1)
    return D


def bit_iter(mask: int):
    """Itera sugli indici dei bit attivi in una bitmask."""
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


def costruisci_grafo_soglia(D: np.ndarray, soglia: float):
    """
    Costruisce il grafo di soglia G_t:
      i ~ j se e solo se D[i, j] <= soglia

    Rappresentazione: lista di bitmask di adiacenza.
    Ogni vertice è adiacente anche a se stesso.
    """
    n = D.shape[0]
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if D[i, j] <= soglia:
                mask |= (1 << j)
        adj[i] = mask
    return adj


def restringi_sottografo(adj_masks, vertex_mask: int):
    """
    Restringe il grafo indotto al sottoinsieme di vertici indicato da vertex_mask.
    """
    return [adj_masks[i] & vertex_mask for i in range(len(adj_masks))]


def bron_kerbosch_pivot(R: int, P: int, X: int, adj_masks, cliques_out):
    """
    Enumerazione delle clique massimali con algoritmo Bron-Kerbosch con pivot.
    Ogni clique è rappresentata come bitmask.
    """
    if P == 0 and X == 0:
        cliques_out.append(R)
        return

    union = P | X
    if union:
        u = next(bit_iter(union))
        candidati = P & ~adj_masks[u]
    else:
        candidati = P

    while candidati:
        v_bit = candidati & -candidati
        v = v_bit.bit_length() - 1

        bron_kerbosch_pivot(
            R | v_bit,
            P & adj_masks[v],
            X & adj_masks[v],
            adj_masks,
            cliques_out
        )

        P ^= v_bit
        X |= v_bit
        candidati ^= v_bit


def clique_massimali(adj_masks, vertex_mask: int):
    """
    Restituisce tutte le clique massimali del sottografo indotto da vertex_mask.
    """
    adj_restricted = restringi_sottografo(adj_masks, vertex_mask)
    out = []
    bron_kerbosch_pivot(0, vertex_mask, 0, adj_restricted, out)
    return out


def clique_cover_minimo_esatto(vertex_mask: int, compat_masks):
    """
    Calcola ESATTAMENTE il minimo numero di clique che ricoprono tutti i vertici
    in vertex_mask nel grafo dato da compat_masks.

    Qui una clique rappresenta un sottoinsieme di diametro <= r
    rispetto alla metrica L_infinito.
    """
    if vertex_mask == 0:
        return 0

    cliques_max = clique_massimali(compat_masks, vertex_mask)

    # Per ogni vertice, memorizza le clique massimali che lo contengono
    vertex_to_cliques = {}
    for v in bit_iter(vertex_mask):
        bit = 1 << v
        lst = []
        for C in cliques_max:
            if C & bit:
                lst.append(C)
        vertex_to_cliques[v] = lst

    @lru_cache(maxsize=None)
    def solve(uncovered_mask: int):
        if uncovered_mask == 0:
            return 0

        # scegli il vertice con meno opzioni
        chosen_v = None
        chosen_options = None
        min_num = None

        for v in bit_iter(uncovered_mask):
            options = []
            for C in vertex_to_cliques[v]:
                C_eff = C & uncovered_mask
                if C_eff != 0:
                    options.append(C_eff)

            if not options:
                return math.inf

            if min_num is None or len(options) < min_num:
                min_num = len(options)
                chosen_v = v
                chosen_options = options
                if min_num == 1:
                    break

        # deduplica e ordina per cardinalità decrescente
        options_unique = list(set(chosen_options))
        options_unique.sort(key=lambda c: c.bit_count(), reverse=True)

        best = math.inf
        for C in options_unique:
            new_uncovered = uncovered_mask & ~C
            val = solve(new_uncovered)
            if val != math.inf:
                best = min(best, 1 + val)

        return best

    return solve(vertex_mask)


def doubling_constant_esatta_linf(X: np.ndarray):
    """
    Calcola ESATTAMENTE la doubling constant lambda(X)
    per un insieme di punti in R^512 con metrica L_infinito.

    Definizione usata:
    ogni sottoinsieme S di diametro <= 2r
    è ricopribile con lambda sottoinsiemi di diametro <= r.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if d != 512:
        raise ValueError("X deve avere shape (n, 512).")

    if n == 0:
        return 0
    if n == 1:
        return 1

    # Matrice delle distanze L_infinito
    D = distanza_linf_matrix(X)

    # Raggi candidati: metà delle distanze positive osservate
    distanze_positive = sorted(
        set(D[i, j] for i in range(n) for j in range(i + 1, n) if D[i, j] > 0)
    )

    if not distanze_positive:
        return 1  # tutti i punti coincidono

    raggi_candidati = [dist / 2.0 for dist in distanze_positive]

    full_mask = (1 << n) - 1
    lambda_star = 1

    for r in raggi_candidati:
        # G_r: due vertici adiacenti se possono stare nello stesso blocco di copertura
        # cioè se la loro distanza L_infinito è <= r
        G_r = costruisci_grafo_soglia(D, r)

        # G_2r: individua i sottoinsiemi S con diametro <= 2r
        G_2r = costruisci_grafo_soglia(D, 2.0 * r)

        # Clique massimali di G_2r
        max_sets = clique_massimali(G_2r, full_mask)

        # Dobbiamo controllare tutti i sottoinsiemi S con diametro <= 2r.
        # Ogni tale S è contenuto in una clique massimale di G_2r.
        visti = set()

        for M in max_sets:
            sub = M
            while sub:
                if sub not in visti:
                    visti.add(sub)
                    cover_num = clique_cover_minimo_esatto(sub, G_r)
                    lambda_star = max(lambda_star, cover_num)
                sub = (sub - 1) & M

    return lambda_star


def doubling_dimension_esatta_linf(X: np.ndarray) -> int:
    """
    Restituisce la doubling dimension esatta rispetto alla norma L_infinito:
        ddim(X) = ceil(log2(lambda(X)))
    """
    lam = doubling_constant_esatta_linf(X)
    if lam <= 1:
        return 0
    return math.ceil(math.log2(lam))