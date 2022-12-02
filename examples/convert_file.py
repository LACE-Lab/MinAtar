import pstats

games = ["breakout"]
suffixes1 = ["qr_dqn"]
suffixes2 = ["0.00001", "0.00005", "0.0001", "0.000005"]
suffixes3 = ["", "16_","2048_","large_"]
for game in games:
    for suffix1 in suffixes1:
        for suffix2 in suffixes2:
            for suffix3 in suffixes3:
                PATH = f"/research/erin/zoshao/results/2022_11_28_{game}_{suffix1}_{suffix3}{suffix2}.txt"
                result_path = f"/research/erin/zoshao/results/2022_11_28_{game}_{suffix1}_{suffix3}{suffix2}.results"
                with open(PATH, 'r') as f:
                    for l in f:
                        sp = l.split()
                        r = sp[8]
                        frame = sp[11]
                        with open(result_path, "a") as d:
                            d.write(str(r)+"\t"+str(frame)+"\n")
                        d.close()
                f.close()
                print(f"/research/erin/zoshao/results/2022_11_28_{game}_{suffix1}_{suffix3}{suffix2}")
        
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