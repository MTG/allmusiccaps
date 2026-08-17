msd_size = 403803
# dt_size = 222405  # size for v1
dt_size = 540442  # size for v2
fs_size = 286083
pse_size = 709536

msd_weight = 0.4
dt_weight = 0.4
fs_weight = 0.1
pse_weight = 0.1

total_size = msd_size + dt_size + fs_size + pse_size

msd_ratio = msd_weight / (msd_size / total_size)
dt_ratio = dt_weight / (dt_size / total_size)
fs_ratio = fs_weight / (fs_size / total_size)
pse_ratio = pse_weight / (pse_size / total_size)

# normalize ratios to sum to 1
sum_ratios = msd_ratio + dt_ratio + fs_ratio + pse_ratio
msd_ratio /= sum_ratios
dt_ratio /= sum_ratios
fs_ratio /= sum_ratios
pse_ratio /= sum_ratios


print(f"dt_ratio: {dt_ratio:.2f}")
print(f"msd_ratio: {msd_ratio:.2f}")
print(f"fs_ratio: {fs_ratio:.2f}")
print(f"pse_ratio: {pse_ratio:.2f}")

print(f"[{dt_ratio:.2f}, {msd_ratio:.2f}, {fs_ratio:.2f}, {pse_ratio:.2f}]")
