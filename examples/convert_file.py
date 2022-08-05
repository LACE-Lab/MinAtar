games = ["space_invaders", "asterix", "breakout", "seaquest", "freeway"]
dims = ["128", "256"]
for game in games:
    for dim in dims:
        PATH = f"/research/erin/zoshao/results/2022_08_01_{game}_{dim}.txt"
        result_path = f"/research/erin/zoshao/results/2022_08_01_{game}_{dim}.results"
        with open(PATH, 'r') as f:
            with open(result_path, "a") as d:
                d.write("Score"+"\t"+"#Frames"+"\n")
            d.close()
            for l in f:
                sp = l.split()
                r = sp[8]
                frame = sp[11]
                with open(result_path, "a") as d:
                    d.write(str(r)+"\t"+str(frame)+"\n")
                d.close()
        f.close()