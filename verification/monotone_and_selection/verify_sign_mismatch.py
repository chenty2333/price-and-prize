from itertools import product
u = lambda S: sum((2,-1)[g] for g in S)
for a in product(range(2), repeat=2):
    B = [{g for g in range(2) if a[g] == i} for i in range(2)]
    subjective = all(
        u(B[i]) >= u(B[1-i])
        or any(u(B[i]) >= u(B[1-i]-{g}) for g in B[1-i] if g == 0)
        or any(u(B[i]-{g}) >= u(B[1-i]) for g in B[i] if g == 1)
        for i in range(2))
    market = abs(len(B[0])-len(B[1])) <= 1
    print(a, subjective, market)
    assert not (subjective and market)
