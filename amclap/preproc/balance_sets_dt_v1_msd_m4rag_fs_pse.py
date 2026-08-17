dt_v1_size = 222405
msd_size = 403803
m4rag_size = 243084
fs_size = 286083
pse_size = 709536

dt_v1_weight = 80 / 3 / 100
msd_weight = 80 / 3 / 100
m4rag_weight = 80 / 3 / 100
fs_weight = 0.1
pse_weight = 0.1

total_size = dt_v1_size + msd_size + m4rag_size + fs_size + pse_size

dt_v1_ratio = dt_v1_weight / (dt_v1_size / total_size)
msd_ratio = msd_weight / (msd_size / total_size)
m4rag_ratio = m4rag_weight / (m4rag_size / total_size)
fs_ratio = fs_weight / (fs_size / total_size)
pse_ratio = pse_weight / (pse_size / total_size)

# normalize ratios to sum to 1
sum_ratios = dt_v1_ratio + msd_ratio + m4rag_ratio + fs_ratio + pse_ratio
dt_v1_ratio /= sum_ratios
msd_ratio /= sum_ratios
m4rag_ratio /= sum_ratios
fs_ratio /= sum_ratios
pse_ratio /= sum_ratios

print(f"dt_v1_ratio: {dt_v1_ratio:.2f}")
print(f"msd_ratio: {msd_ratio:.2f}")
print(f"m4rag_ratio: {m4rag_ratio:.2f}")
print(f"fs_ratio: {fs_ratio:.2f}")
print(f"pse_ratio: {pse_ratio:.2f}")

print(
    f"[{dt_v1_ratio:.2f}, {msd_ratio:.2f}, {m4rag_ratio:.2f}, {fs_ratio:.2f}, {pse_ratio:.2f}]"
)
