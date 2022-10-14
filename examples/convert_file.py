games = ["breakout"]
suffixes1 = ["ac"]
suffixes2 = ["decoder", "no_brick_v_0.00001", "no_brick_v_0.0001", "no_type_matrix", "no_type_matrix_0.00001", "no_type_vec_0.00001", "no_type_vec_0.0001"]
for game in games:
    for suffix1 in suffixes1:
        for suffix2 in suffixes2:
            PATH = f"/research/erin/zoshao/results/2022_10_12_{game}_{suffix1}_{suffix2}.txt"
            result_path = f"/research/erin/zoshao/results/2022_10_12_{game}_{suffix1}_{suffix2}.results"
            with open(PATH, 'r') as f:
                for l in f:
                    sp = l.split()
                    r = sp[8]
                    frame = sp[11]
                    with open(result_path, "a") as d:
                        d.write(str(r)+"\t"+str(frame)+"\n")
                    d.close()
            f.close()