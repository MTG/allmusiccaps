dt_v1_size = 222405
msd_size = 403803
m4rag_size = 243084

dt_v1_weight = 1 / 3
msd_weight = 1 / 3
m4rag_weight = 1 / 3

total_size = dt_v1_size + msd_size + m4rag_size

dt_v1_ratio = dt_v1_weight / (dt_v1_size / total_size)
msd_ratio = msd_weight / (msd_size / total_size)
m4rag_ratio = m4rag_weight / (m4rag_size / total_size)

# normalize ratios to sum to 1
sum_ratios = dt_v1_ratio + msd_ratio + m4rag_ratio
dt_v1_ratio /= sum_ratios
msd_ratio /= sum_ratios
m4rag_ratio /= sum_ratios

print(f"dt_v1_ratio: {dt_v1_ratio:.2f}")
print(f"msd_ratio: {msd_ratio:.2f}")
print(f"m4rag_ratio: {m4rag_ratio:.2f}")

print(f"[{dt_v1_ratio:.2f}, {msd_ratio:.2f}, {m4rag_ratio:.2f}]")
