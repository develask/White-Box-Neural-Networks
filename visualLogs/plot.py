import matplotlib
matplotlib.use('TkAgg')
import json
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.ticker as ticker
from matplotlib import colors
import numpy as np
import time
import math

COLORS = [
	{
			'name': 'Deep Orange',
			'color': '#ff6f42',
			'active': '#ca4a06',
			'disabled': '#f2cbba'
		},
		{
			'name': 'Google Blue',
			'color': '#4184f3',
			'active': '#3a53c5',
			'disabled': '#cad8fc'
		},
		{
			'name': 'Google Yellow',
			'color': '#f4b400',
			'active': '#db9200',
			'disabled': '#f7e8b0'
		},
		{
			'name': 'Google Green',
			'color': '#0f9d58',
			'active': '#488046',
			'disabled': '#c2e1cc'
		},
		{
			'name': 'Pink',
			'color': '#ef6191',
			'active': '#ca1c60',
			'disabled': '#e9b9ce'
		},
		{
			'name': 'Purple',
			'color': '#aa46bb',
			'active': '#5c1398',
			'disabled': '#d7bce6'
		},
		{
			'name': 'Teal',
			'color': '#00abc0',
			'active': '#47828e',
			'disabled': '#c2eaf2'
		},
		{
			'name': 'Lime',
			'color': '#9d9c23',
			'active': '#7f771d',
			'disabled': '#f1f4c2'
		},
	{
			'name': 'Google Red',
			'color': '#db4437',
			'active': '#8f2a0c',
			'disabled': '#e8c6c1'
		},
		{
			'name': 'Indigo',
			'color': '#5b6abf',
			'active': '#3e47a9',
			'disabled': '#c5c8e8'
		},
		{
			'name': 'Deep Teal',
			'color': '#00786a',
			'active': '#2b4f43',
			'disabled': '#bededa'
		},
		{
			'name': 'Deep Pink',
			'color': '#c1175a',
			'active': '#75084f',
			'disabled': '#de8cae'
		},
		{
			'name': 'Gray',
			'color': '#9E9E9E',
			'active': '#424242',
			'disabled': '#F5F5F5'
		}
]

def to_rgb(c):
	r = int(c[1:3],16)
	g = int(c[3:5],16)
	b = int(c[5:7],16)
	return r,g,b


class Plotter():
	def __init__(self, files_to_read):
		self.files_to_read = files_to_read

		show_folder =  len(set([file_to_read.split('/')[-2] for file_to_read in files_to_read])) > 1

		self.logs = {}
		for file_to_read in files_to_read:
			format_ = file_to_read.split('/')[-1].split('_')[0]
			if format_ != self.type:
				continue

			name = '_'.join(file_to_read.split('/')[-1].split('_')[1:])[:-4]
			if show_folder:
				name = name + ' ('+file_to_read.split('/')[-2]+')'

			my_logs = self.read_file(file_to_read)
			self.logs[name] = my_logs


		if len(list(self.logs)) > 0:
			self.plot(self.logs)

	def read_file(self, file):
		return None

	def plot(self):
		print("No plot function")

class Histogram(Plotter):
	def __init__(self, files_to_read):
		self.type = 'hist'
		super(Histogram, self).__init__(files_to_read)

	def read_file(self, file_to_read):
		with open(file_to_read) as f:
			logs = f.read().splitlines()
		return [ ( float(x.split('\t')[1]), json.loads(x.split('\t')[2].replace('(','[').replace(')',']')) ) for x in logs]

	def plot(self, logs_dict):

		num_cols = math.ceil(math.sqrt(len(list(logs_dict))))
		if len(list(logs_dict)) > num_cols*(num_cols-1):
			num_rows = num_cols
		else:
			num_rows = num_cols-1

		# fig = plt.figure()
		fig = plt.figure(figsize=(num_cols*4, num_rows*4), dpi=80)
		num = 1
		self.axes = []
		for name in logs_dict:
			logs = logs_dict[name]
			color = COLORS[(num-1) % len(COLORS)]
			def polygon_under_graph(xlist, ylist):
					return [(xlist[0], 0.)] + list(zip(xlist, ylist)) + [(xlist[-1], 0.)]

			def color_(i):
				back = to_rgb(color['active'])
				front = to_rgb(color['disabled'])
				porcent = i / len(logs)
				def take(min_, max_, porcent):
					return ( min_ + ((max_-min_)*porcent) ) / 255
				return take(back[0], front[0], porcent), take(back[1], front[1], porcent), take(back[2], front[2], porcent), 0.8


			ax = fig.add_subplot(num_rows, num_cols, num, projection='3d')
			self.axes.append(ax)

			temps = []
			zs = range(len(logs))

			my_mod = int(len(logs)/10)
			if my_mod == 0:
				my_mod = 1

			#filer
			logs, zs = zip(*[ (x,i) for x,i in zip(logs, zs) if i%my_mod==0])

			val_max = 0
			num_vals = np.sum(np.asarray([x[1] for x in logs])[:,0,1])
			left_lim = 0
			right_lim = 0

			for (width, bar), z in zip(logs, zs):
				xs = [x[0]+width/2 for x in bar]
				ys = [x[1]/(width*num_vals) for x in bar]
				if max(ys) > val_max:
					val_max = max(ys)
				if min(xs) < right_lim:
					right_lim = min(xs)
				if max(xs) > left_lim:
					left_lim = max(xs)

				temps.append(polygon_under_graph(xs, ys))

				# ax.plot(xs, ys, z, zdir='y', color=(1,1,1), alpha=1, linewidth=1)

			poly = PolyCollection(temps, facecolors=[color_(x) for x in range(len(logs))], edgecolors=(1,1,1,0.8))
			ax.add_collection3d(poly, zs=zs, zdir='y')

			ax.set_xlabel('value')
			ax.set_ylabel('mini-batch')
			ax.set_xlim(right_lim, left_lim)
			ax.set_ylim(zs[-1], zs[0])
			ax.set_zlim(0, val_max)
			plt.title(name)
			num += 1

		def update(val):
			num =1
			new_axes = []
			for ax, name in zip(self.axes, list(logs_dict)):
				logs = logs_dict[name]
				color = COLORS[(num -1) % len(COLORS)]
				ax.remove()
				ax = fig.add_subplot(num_rows, num_cols, num, projection='3d')
				new_axes.append(ax)
				temps = []
				zs = range(len(logs))

				my_mod = int(len(logs)/int(val))
				if my_mod == 0:
					my_mod = 1

				#filer
				logs, zs = zip(*[ (x,i) for x,i in zip(logs, zs) if i%(my_mod)==0])

				def color_(i):
					back = to_rgb(color['active'])
					front = to_rgb(color['disabled'])
					porcent = i / len(logs)
					def take(min_, max_, porcent):
						return ( min_ + ((max_-min_)*porcent) ) / 255
					return take(back[0], front[0], porcent), take(back[1], front[1], porcent), take(back[2], front[2], porcent), 0.8

				val_max = 0
				num_vals = np.sum(np.asarray([x[1] for x in logs])[:,0,1])
				left_lim = 0
				right_lim = 0

				for (width, bar), z in zip(logs, zs):
					xs = [x[0]+width/2 for x in bar]
					ys = [x[1]/(width*num_vals) for x in bar]
					if max(ys) > val_max:
						val_max = max(ys)
					if min(xs) < right_lim:
						right_lim = min(xs)
					if max(xs) > left_lim:
						left_lim = max(xs)
					temps.append(polygon_under_graph(xs, ys))

				poly = PolyCollection(temps, facecolors=[color_(x) for x in range(len(logs))], edgecolors=(1,1,1,0.8))
				ax.add_collection3d(poly, zs=zs, zdir='y')

				ax.set_xlabel('value')
				ax.set_ylabel('mini-batch')
				ax.set_xlim(right_lim, left_lim)
				ax.set_ylim(zs[-1], zs[0])
				ax.set_zlim(0, val_max)
				plt.title(name)
				num+=1
			self.axes = new_axes

		fig.subplots_adjust(top=0.99,bottom=0.1, left=0.01, right=0.99)

		axcolor = 'lightgoldenrodyellow'
		axsmooth = plt.axes([0.25, 0.025, 0.60, 0.03], facecolor=axcolor)
		samp = Slider(axsmooth, 'samples', 5, 50, valinit=10, valstep=1)
		samp.on_changed(update)

		plt.show()

class Scalar(Plotter):
	def __init__(self, files_to_read):
		self.type = 'scalar'
		super(Scalar, self).__init__(files_to_read)

	def read_file(self, file_to_read):
		with open(file_to_read) as f:
			logs = f.read().splitlines()
		return [ (float(x.split('\t')[0]), float(x.split('\t')[1])) for x in logs]


	def plot(self, logs_dict, smooth_constant=1):
												# ['relative-time', 'real-time', 'iteration']

		names = list(logs_dict)
		names = sorted(names, key=lambda x: len(logs_dict[x]), reverse=True)
		mode = 'relative-time'
		fig, self.ax = plt.subplots()
		if mode == 'relative-time':
			time_min = min([logs_dict[name][0][0] for name in logs_dict])

			formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))
			self.ax.xaxis.set_major_formatter(formatter)

		if mode == 'real-time':
			formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime("%b %d %H:%M:%S", time.gmtime(ms)))
			self.ax.xaxis.set_major_formatter(formatter)

		def mean(list_):
			return sum(list_)/len(list_)

		def to_y(x, max_x = 700, max_y = 30):
			if x > max_x:
				x = max_x
			return max_y/(max_x**2) * x**2 - max_y/(max_x**2)

		def smoothed_list(values, smooth_constant):
			smooth_dist = int(smooth_constant * to_y(len(values)))
			return [ mean(values[(i-smooth_dist if i-smooth_dist>=0 else 0):i+smooth_dist+1]) for i in range(len(values))]

		def update(val):
			mode = radio.value_selected
			if mode == 'relative-time':
				time_min = min([logs_dict[name][0][0] for name in names])
				formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%H:%M:%S', time.gmtime(ms)))
				self.ax.xaxis.set_major_formatter(formatter)

			if mode == 'real-time':
				formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime("%b %d %H:%M:%S", time.gmtime(ms)))
				self.ax.xaxis.set_major_formatter(formatter)

			if mode == 'iteration':
				self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))


			i = 0
			for name in names:
				logs = logs_dict[name]
				times, values = zip(*logs)
				time_min = times[0]
				if mode == 'relative-time':
					times = [x-time_min for x in times]
				if mode == 'iteration':
					times = list(range(len(times)))
				smooth_values = smoothed_list(values, samp.val)
				backs[i].set_ydata(values)
				backs[i].set_xdata(times)
				i+=1

			i = 0
			for name in names:
				logs = logs_dict[name]
				times, values = zip(*logs)
				time_min = times[0]
				if mode == 'relative-time':
					times = [x-time_min for x in times]
				if mode == 'iteration':
					times = list(range(len(times)))
				smooth_values = smoothed_list(values, samp.val)
				legends[i].set_ydata(smooth_values)
				legends[i].set_xdata(times)
				i+=1

			self.ax.relim()
			self.ax.autoscale_view()

		legends = []
		backs = []
		c = 0
		for name in names:
			logs = logs_dict[name]
			times, values = zip(*logs)
			time_min = times[0]
			if mode == 'relative-time':
				times = [x-time_min for x in times]
			if mode == 'iteration':
				times = list(range(len(times)))
			color = COLORS[c % len(COLORS)]
			smooth_values = smoothed_list(values, smooth_constant)
			b, = plt.plot(times,values, '-', color=color['disabled'])
			backs.append(b)
			c+=1
		c = 0
		for name in names:
			logs = logs_dict[name]
			times, values = zip(*logs)
			time_min = times[0]
			if mode == 'relative-time':
				times = [x-time_min for x in times]
			if mode == 'iteration':
				times = list(range(len(times)))
			color = COLORS[c % len(COLORS)]
			smooth_values = smoothed_list(values, smooth_constant)
			leg, = plt.plot(times,smooth_values, '-', color=color['active'], label=name)
			legends.append(leg)
			c+=1

		# plt.title("Scalar over time")

		plt.legend(handles=legends)

		plt.gcf().autofmt_xdate()
		plt.grid()
		fig.subplots_adjust(bottom=0.25)

		axcolor = 'lightgoldenrodyellow'
		axsmooth = plt.axes([0.35, 0.025, 0.5, 0.03], facecolor=axcolor)
		samp = Slider(axsmooth, 'Smooth', 0.01, 2.0, valinit=1.0)
		samp.on_changed(update)

		rax = plt.axes([0.025, 0.025, 0.2, 0.1], facecolor=axcolor)
		radio = RadioButtons(rax, ['relative-time', 'real-time', 'iteration'], active=0)

		radio.on_clicked(update)

		plt.show()

class Image(Plotter):
	def __init__(self, files_to_read):
		self.type = 'image'
		super(Image, self).__init__(files_to_read)

	def read_file(self, file_to_read):
		data = np.load(file_to_read)
		return data['times'], data['values']

	def plot(self, logs_dict):
		num_cols = math.ceil(math.sqrt(len(list(logs_dict))))
		if len(list(logs_dict)) > num_cols*(num_cols-1):
			num_rows = num_cols
		else:
			num_rows = num_cols-1

		names = list(logs_dict)

		_min = 0
		_max = 0
		for title in names:
			new_min = np.min(logs_dict[title][1])
			new_max = np.max(logs_dict[title][1])
			if new_min < _min:
				_min = new_min
			if new_max > _max:
				_max = new_max

		# fig = plt.figure()
		fig, axeslist = plt.subplots(ncols=num_cols, nrows=num_rows)
		if num_cols * num_rows == 1:

			class MyC():
				def __init__(self, axis):
					self.axis = axis
				def ravel(self):
					return [self.axis]
			axeslist = MyC(axeslist)
		for ind,title in zip(range(len(names)), names):
			im = axeslist.ravel()[ind].imshow(logs_dict[title][1][-1,...], cmap='hot', vmin = _min, vmax = _max, interpolation='nearest', aspect='auto')
			axeslist.ravel()[ind].set_title(title)
			axeslist.ravel()[ind].set_axis_off()
		for last_idx in range(ind+1, num_cols*num_rows):
			axeslist.ravel()[last_idx].remove()
		plt.tight_layout() # optional

		fig.subplots_adjust(top=0.95,bottom=0.15, left=0.1, right=0.8)

		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax=cbar_ax)

		def update(val):
			for ind,title in zip(range(len(names)), names):
				if width_b.value_selected == 'original':
					im = axeslist.ravel()[ind].imshow(logs_dict[title][1][int((logs_dict[title][1].shape[0]-1)*samp.val),...], cmap=radio.value_selected, vmin = _min, vmax = _max)
				else:
					im = axeslist.ravel()[ind].imshow(logs_dict[title][1][int((logs_dict[title][1].shape[0]-1)*samp.val),...], cmap=radio.value_selected, vmin = _min, vmax = _max, interpolation='nearest', aspect='auto')

			fig.colorbar(im, cax=cbar_ax)

		axcolor = 'lightgoldenrodyellow'
		axsmooth = plt.axes([0.35, 0.025, 0.55, 0.03], facecolor=axcolor)
		samp = Slider(axsmooth, 'time', 0, 1, valinit=1)
		samp.on_changed(update)

		rax = plt.axes([0.025, 0.025, 0.1, 0.1], facecolor=axcolor)
		radio = RadioButtons(rax, ('hot', 'jet'), active=0)
		radio.on_clicked(update)

		rax2 = plt.axes([0.15, 0.025, 0.1, 0.1], facecolor=axcolor)
		width_b = RadioButtons(rax2, ('scaled', 'original'), active=0)
		width_b.on_clicked(update)

		plt.show()

def main(args=None):
	"""The main routine."""
	if args is None:
		args = sys.argv[1:]

	files_to_read = args

	for C in [Scalar, Histogram, Image]:
		c = C(files_to_read)


if __name__ == '__main__':
	main()



