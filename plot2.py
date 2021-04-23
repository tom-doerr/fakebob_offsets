#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
import re
import argparse
import os

#if len(sys.argv) < 2:
#    filter_term = ''
#else:
#    filter_term = sys.argv[1]


SAVE_FOLDER = './figures/'


parser = argparse.ArgumentParser()
parser.add_argument('filter_term', type=str)
parser.add_argument('--save_as', type=str)
parser.add_argument('--label_language', type=str)



args = parser.parse_args()

filter_term = args.filter_term






from offset_scores.out_dict import offset_dict
print(offset_dict)

avg_scores = []
max_min_diff_list = []
list_var = []
for key in offset_dict.keys():
#for re.search(key, offset_dict.keys()):
    #if filter_term not in key:
    if not re.search(filter_term, key):
        continue
    average_score = sum(offset_dict[key])/len(offset_dict[key])
    avg_scores.append(average_score)
    print("average_score:", average_score)
    plt.plot(offset_dict[key], label=key)
    max_min_diff = max(offset_dict[key]) - min(offset_dict[key])
    print("max_min_diff:", max_min_diff)
    max_min_diff_list.append(max_min_diff)
    var = sum([(e - average_score)**2 for e in offset_dict[key]])/(len(offset_dict[key]) - 1)
    list_var.append(var)


print('===================\n\n')
average_over_audios = sum(avg_scores)/len(avg_scores)
print("average_over_audios:", average_over_audios)
average_over_audios_max_min = sum(max_min_diff_list)/len(max_min_diff_list)
print("average_over_audios_max_min:", average_over_audios_max_min)
average_var = sum(list_var)/len(list_var)
print("average_var:", average_var)


#print(f'{args.save_as.replace(".png", "")}, {average_over_audios}, {average_over_audios_max_min}, {average_var}')
if args.save_as:
    print(f'{args.save_as.replace(".png", "")}, {average_over_audios}, {average_var}')

if args.label_language == 'german':
    plt.xlabel('Verschiebung in Samples')
    plt.ylabel('Bewertung')
elif args.label_language == 'english':
    plt.xlabel('Offset in Samples')
    plt.ylabel('Score')

if args.save_as:
    save_path = os.path.join(SAVE_FOLDER, args.save_as)
    plt.savefig(save_path)
else:
    plt.legend()
    plt.show()





#with open(OFFSET_SCORES_DIR + offset_scores_file, 'r') as f:
#    offset_list = f.read()[1:-1].split(', ')
#    offset_list = [float(e) for e in offset_list]
#
#
#print(f'mean: {sum(offset_list)/len(offset_list)}')
#plt.plot(offset_list)
#plt.show()
#

