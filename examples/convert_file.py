games = ["breakout"]
suffixes1 = ["ac"]
suffixes2 = ["0.01","0.001","0.0001", "0.00001", "0.000001"]
suffixes3 = ["_decoder"]
for game in games:
    for suffix1 in suffixes1:
        for suffix2 in suffixes2:
            for suffix3 in suffixes3:
                PATH = f"/research/erin/zoshao/results/2022_10_07_{game}_{suffix1}_{suffix2}{suffix3}.txt"
                result_path = f"/research/erin/zoshao/results/2022_10_07_{game}_{suffix1}_{suffix2}{suffix3}.results"
                with open(PATH, 'r') as f:
                    for l in f:
                        sp = l.split()
                        r = sp[8]
                        frame = sp[11]
                        with open(result_path, "a") as d:
                            d.write(str(r)+"\t"+str(frame)+"\n")
                        d.close()
                f.close()