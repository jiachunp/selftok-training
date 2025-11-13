import re
import os
import json
import pandas as pd
import yaml
import argparse


def parse_yaml(fp):
    with open(fp, "r") as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return y


def parse_timeline(filepath):
    # 算子名->算子类型映射
    cate2name = parse_yaml(os.path.join(os.path.dirname(__file__), "mapping.yml"))
    name2cate = {}
    for cate_name, op_names in cate2name.items():
        for op_name in op_names:
            name2cate[op_name] = cate_name

    filedir, filename = os.path.split(filepath)
    print("start analysis {}".format(filepath))
    with open(filepath) as fn:
        ops = json.load(fn)

    ops = ops["traceEvents"]
    print("find {} events".format(len(ops)))

    cat_ = [x["cat"] if "cat" in x.keys() else None for x in ops]
    cat_key = list(set(cat_))

    out_filenames = []

    for key in cat_key:
        if key is None:
            cat_this = [x for x in ops if "cat" not in x.keys()]
        else:
            cat_this = [x for x in ops if "cat" in x.keys() and x["cat"] == key]

        print("")
        print("find cat={}, cnt={}".format(key, len(cat_this)))

        if key in ["Kernel", "kernel"]:
            stream7 = [x for x in cat_this if x["args"]["stream"] == 7]
            not_stream7 = [x for x in cat_this if x["args"]["stream"] != 7]  # nccl, pass
            tot_time = sum([x["dur"] for x in stream7]) / 1000
            print("stream7: total_time={}ms".format(tot_time))

            ops_groupby_op_name = {}
            ops_groupby_cate = {}

            for x in stream7:
                if x["name"] not in ops_groupby_op_name.keys():
                    ops_groupby_op_name[x["name"]] = []  # name, dur#
                ops_groupby_op_name[x["name"]].append(x["dur"])

                op_name_this = x["name"][5:] if x["name"].startswith("void ") else x["name"]
                real_cate_name = []
                for name_pattern, cate_name in name2cate.items():
                    ret = re.match(name_pattern, op_name_this)
                    if ret is not None:
                        real_cate_name.append(cate_name)

                if len(real_cate_name) > 1:
                    print("ERROR: op_name: {}, has many pattern: {}".format(x["name"], real_cate_name))
                    exit()
                elif len(real_cate_name) == 1:
                    real_cate_name = real_cate_name[0]
                else:
                    real_cate_name = x["name"]

                if real_cate_name not in ops_groupby_cate.keys():
                    ops_groupby_cate[real_cate_name] = []  # name, dur#
                ops_groupby_cate[real_cate_name].append(x["dur"])

            for save_tag, init_data in [["init", ops_groupby_op_name], ["cate", ops_groupby_cate]]:
                ops_list = []
                for k, v in init_data.items():
                    min_ = min(v)
                    max_ = max(v)
                    tot_ = sum(v)
                    mean_ = tot_ / len(v)
                    cnt_ = len(v)
                    percent_ = tot_ / 1000.0 / tot_time
                    ops_list.append([k, cnt_, tot_ / 1000, mean_ / 1000, min_ / 1000, max_ / 1000, percent_])

                ops_list.sort(key=lambda x: x[-1], reverse=True)
                df = pd.DataFrame(ops_list, columns=["op_name", "cnt", "sum_t", "avg_t", "min_t", "max_t", "percent"])

                save_name = "op_statistic_" + save_tag + "_" + filename.replace(".json", ".xlsx")
                save_name = os.path.join(filedir, save_name)
                out_filenames.append(save_name)
                df.to_excel(save_name, index=False)

            summary = [[x["name"], x["dur"] / 1000.0] for x in stream7]
            df = pd.DataFrame(
                summary,
                columns=[
                    "op_name",
                    "dur",
                ],
            )

            save_name = "op_summary_" + filename.replace(".json", ".xlsx")
            save_name = os.path.join(filedir, save_name)
            out_filenames.append(save_name)
            df.to_excel(save_name, index=False)

    return out_filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gpu timeline parser")
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="filepath",
    )
    args, _ = parser.parse_known_args()

    if os.path.exists(args.filepath):
        parse_timeline(args.filepath)
