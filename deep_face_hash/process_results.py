import dill
import pickle
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from os import path

# test_size = 100

def process_accuracies():
	test_size = 100
	in_file = path.abspath(path.join(path.dirname(__file__), "data", "accuracy_" + str(test_size) + ".p"))
	accuracies = pickle.load( open( in_file, "rb" ) )

	print(accuracies)

	for ws, hs_szs in accuracies.iteritems():
		for hs, scores in hs_szs.iteritems():
			rank_list = []
			acc_list = []
			for scoreset in scores:
				for rank, acc in scoreset.iteritems():
					if rank == 1:
						rank += 0.08
					elif rank == 10:
						rank -= 0.08
					rank_list.append(rank)
					acc_list.append(acc)
			title = "Window size - " + str(ws) + " - Hash size - " + str(hs)


			plt.rc('font', weight='bold')
			plt.rc('xtick.major', size=5, pad=7)
			plt.rc('xtick', labelsize=15)

			fig = plt.figure()
			ax = fig.add_subplot(111)

			font = FontProperties()

			# sizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
			# 		 'x-large', 'xx-large']

			font.set_weight('bold')
			font.set_size('large')

			plt.plot(rank_list, acc_list, 'ro')

			for x,y in zip(rank_list, acc_list):
				ax.annotate('%s' % y, xy=(x,y+0.15), textcoords='data', fontproperties=font)

			plt.axis([1, 10, 0, 20])
			plt.xlabel('Rank', fontproperties=font)
			plt.ylabel('Accuracy (%)', fontproperties=font)
			plt.title(title, fontproperties=font)
			# plt.show()

			fig_title = str(test_size) + "_" + str(ws) + "_" + str(hs) + ".eps"
			fig_file = path.abspath(path.join(path.dirname(__file__), "data", "figures" , fig_title))
			fig.savefig(fig_file)


def process_feat_map_accuracies():
	test_size = 100
	in_file = path.abspath(path.join(path.dirname(__file__), "data", "old_feat_map_accuracy_" + str(test_size) + ".p"))
	accuracies = pickle.load( open( in_file, "rb" ) )

	print(accuracies)

	rank_list = []
	acc_list = []
	for rank, acc in accuracies.iteritems():
		if rank == 1:
			rank += 0.08
		elif rank == 10:
			rank -= 0.08
		rank_list.append(rank)
		acc_list.append(acc)
	title = "Feature maps accuracy without viola-jones"


	plt.rc('font', weight='bold')
	plt.rc('xtick.major', size=5, pad=7)
	plt.rc('xtick', labelsize=15)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	font = FontProperties()

	# sizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
	# 		 'x-large', 'xx-large']

	font.set_weight('bold')
	font.set_size('large')

	plt.plot(rank_list, acc_list, 'ro')

	for x,y in zip(rank_list, acc_list):
		ax.annotate('%s' % y, xy=(x,y+0.15), textcoords='data', fontproperties=font)

	plt.axis([1, 10, 0, 20])
	plt.xlabel('Rank', fontproperties=font)
	plt.ylabel('Accuracy (%)', fontproperties=font)
	plt.title(title, fontproperties=font)
	plt.show()

	fig_title = str(test_size) + "_" + "feature_maps_old" + ".eps"
	fig_file = path.abspath(path.join(path.dirname(__file__), "data", "figures" , fig_title))
	fig.savefig(fig_file)

if __name__ == '__main__':
	process_accuracies()
	# process_feat_map_accuracies()