pairs = [(0, 10), (1, 15), (2, 7)]

pairs_m_10 = filter(lambda x: x[1] <= 10, pairs)

# for i in pairs_m_10:
    # print(i)

min_val = min(pairs, key=lambda t: t[1])
# print(min_val)

pairs.pop(0)
print(pairs)