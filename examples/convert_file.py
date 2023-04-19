import pstats

games = ["test", "test2"]
suffixes1 = ["qr_dqn", "qr_dqn_com"]
suffixes2 = ["a_0.00001", "a_0.0001", "w_0.0001", "w_0.001"]
# suffixes3 = ["a", "w"]
for game in games:
    for suffix1 in suffixes1:
        for suffix2 in suffixes2:
            # for suffix3 in suffixes3:
            # PATH = f"/research/erin/zoshao/results/2023_01_25_{game}_{suffix1}_{suffix2}.txt"
            # result_path = f"/research/erin/zoshao/results/2023_01_25_{game}_{suffix1}_{suffix2}.results"
            PATH = "test.txt"
            result_path = "test.results"
            with open(PATH, 'r') as f:
                for l in f:
                    sp = l.split()
                    r = sp[8]
                    frame = sp[11]
                    print(r, frame)
                    with open(result_path, "a") as d:
                        d.write(str(r)+"\t"+str(frame)+"\n")
                    d.close()
            f.close()
            print(f"/research/erin/zoshao/results/2022_01_25_{game}_{suffix1}_{suffix2}")
        
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