import pstats

suffixes1 = ["cart_pole"]
suffixes2 = ["0.001", "0.01", "0.01_no_hardcode", "0.01_no_hardcode_h1", "0.1", "0.1_no_hardcode", "0.1_no_hardcode_h1", "a2_0.001", "a2_0.01", "a2_0.01_h1", "a2_0.1", "a2_0.1_h1", "no_rollout", "perfect", "perfect_h1"]
# suffixes3 = ["a", "w"]

for suffix1 in suffixes1:
    for suffix2 in suffixes2:
        # for suffix3 in suffixes3:
        PATH = f"/research/erin/zoshao/results/2023_04_26_{suffix1}_{suffix2}.txt"
        result_path = f"/research/erin/zoshao/results/2023_04_26_{suffix1}_{suffix2}.results"
        with open(PATH, 'r') as f:
            for l in f:
                sp = l.split()
                r = sp[8]
                frame = sp[11]
                with open(result_path, "a") as d:
                    d.write(str(r)+"\t"+str(frame)+"\n")
                d.close()
        f.close()
        print(f"/research/erin/zoshao/results/2022_04_26_{suffix1}_{suffix2}")
        
# file = open('/research/erin/zoshao/results/2022_10_28_profile_results_ac_0.00001_no_opt_tottime.txt', 'w')
# profile = pstats.Stats('/research/erin/zoshao/results/2022_10_28_profile_results_ac_0.00001_no_opt', stream=file)
# profile.sort_stats('tottime') # Sorts the result according to the supplied criteria
# profile.print_stats(30) # Prints the first 15 lines of the sorted report
# file.close()

# file = open('/research/erin/zoshao/results/2022_10_28_profile_results_ac_0.01_no_opt_tottime.txt', 'w')
# profile = pstats.Stats('/research/erin/zoshao/results/2022_10_28_profile_results_ac_0.01_no_opt', stream=file)
# profile.sort_stats('tottime') # Sorts the result according to the supplied criteria
# profile.print_stats(30) # Prints the first 15 lines of the sorted report
# file.close()