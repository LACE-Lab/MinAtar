games = ["space_invaders", "asterix", "breakout", "seaquest", "freeway"]
for game in games:
    PATH = f"/research/erin/zoshao/results/2022_08_03_{game}.txt"
    result_path = f"2022_08_03_{game}.results"
    with open(PATH, 'r') as f:
        with open(result_path, "a") as d:
            d.write("Score"+"\t"+"#Frames"+"\n")
        d.close()
        for l in f:
            sp = l.split()
            print(sp)
            r = sp[8]
            frame = sp[11]
            print(r, frame)
            with open(result_path, "a") as d:
                d.write(str(r)+"\t"+str(frame)+"\n")
            d.close()
    f.close()