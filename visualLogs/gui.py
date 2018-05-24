from tkinter import *
from tkinter import ttk, filedialog

from os import listdir
from os.path import isfile, join

import sys
from . import plot


class WBNN_Plots(Tk):
	def __init__(self):
		super(WBNN_Plots, self).__init__()
		self.grid_columnconfigure(0, weight=1)
		self.grid_rowconfigure(0,weight=1)

		self.title("WBNN Plots")

		self.minsize(700, 300)

		self.menu()

		self.file_folders()
		self.plots_side()

	def menu(self):
		menubar = Menu(self)

		# create a pulldown menu, and add it to the menu bar
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="Open", command=lambda: self.new_file_folder(filedialog.askdirectory()))
		filemenu.add_command(label="New Plot Area", command=lambda: self.new_plot_area())
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=self.quit)
		menubar.add_cascade(label="File", menu=filemenu)

		# display the menu
		self.config(menu=menubar)

	def file_folders(self):
		self.file_manager = ttk.Notebook(self, padding=(5, 5, 12, 0))
		self.file_manager.grid_columnconfigure(0, weight=1)
		self.file_manager.grid_rowconfigure(5, weight=1)
		self.file_manager.grid(column=0, row=0, sticky=(N,W,E,S))

		self.folders = []

	def new_file_folder(self, site="."):

		onlyfiles = [f for f in listdir(site) if isfile(join(site, f)) and f.split('_')[0] in ['hist', 'scalar', 'image']]

		onlyfiles = sorted(onlyfiles)

		frame = ttk.Notebook(self, padding=(5, 5, 12, 0))
		frame.grid(column=0, row=0, sticky=(N,W,E,S))
		frame.grid_columnconfigure(0, weight=1)
		frame.grid_rowconfigure(5, weight=1)

		lbl_name = ttk.Label(frame, text="folder:\n"+(site[-49:] if len(site)<=49 else "..."+site[-45:] ))
		lbl_name.grid(column=0, row=0, padx=0, pady=5, sticky=(N,S,E,W))

		c_files = StringVar(value=onlyfiles)

		lbox = Listbox(frame, listvariable=c_files, height=30, width=40)
		lbox.grid(column=0, row=1, rowspan=6, sticky=(N,S,E,W))

		lbox.bind('<<ListboxSelect>>', lambda x: self.append_file_to_plot_area(lbox.curselection()))

		for i in range(1,len(onlyfiles),2):
			lbox.itemconfigure(i, background='#f0f0ff')

		exit_folder = ttk.Button(frame, text='close', command=lambda : self.delete_file_folder(site), default='active')
		exit_folder.grid(column=0, row=20, sticky=S)

		self.file_manager.add(frame, text=str(len(self.folders)+1))

		self.file_manager.select(len(self.folders))

		self.folders.append({
			'folder': site,
			'frame': frame,
			'files': onlyfiles
		})

	def delete_file_folder(self, folder_dir):
		if len(self.folders) == 1:
			return
		idx = -1
		for i, folder in enumerate(self.folders):
			if folder['folder'] == folder_dir:
				idx = i
		self.file_manager.forget(idx)
		del self.folders[idx]
		self.file_manager.select(len(self.folders)-1)

		for i in range(len(self.folders)):
			self.file_manager.tab(self.folders[i]['frame'], text = str(i+1))

	def append_file_to_plot_area(self, file_idx):
		if len(file_idx)==0:
			return
		file_idx = file_idx[0]
		idx_of_folders = self.file_manager.index(self.file_manager.select())
		folder_ = self.folders[idx_of_folders]['folder']
		file_ = self.folders[idx_of_folders]['files'][file_idx]

		idx_of_plots = self.plots_manager.index(self.plots_manager.select())

		if (folder_, file_) in self.plots_confs[idx_of_plots]['files']:
			return

		self.plots_confs[idx_of_plots]['files'].append((folder_, file_))
		self.plots_confs[idx_of_plots]['lbox'].insert('end', file_+" ("+folder_+")" )

		for i in range(1,len(self.plots_confs[idx_of_plots]['files']),2):
			self.plots_confs[idx_of_plots]['lbox'].itemconfigure(i, background='#f0f0ff')


	def plots_side(self):
		self.plots_manager = ttk.Notebook(self, padding=(5, 5, 12, 0))
		self.plots_manager.grid(column=1, row=0, sticky=(N,W,E,S))
		self.plots_manager.grid_columnconfigure(0, weight=1)
		self.plots_manager.grid_rowconfigure(5, weight=1)

		self.plots_confs = []

		self.new_plot_area()

	def new_plot_area(self):

		frame = ttk.Notebook(self, padding=(5, 5, 12, 0))
		frame.grid(column=0, row=0, sticky=(N,W,E,S))
		frame.grid_columnconfigure(0, weight=1)
		frame.grid_rowconfigure(5, weight=1)

		lbox = Listbox(frame, listvariable=None, height=30, width=40)
		lbox.grid(column=0, row=1, rowspan=6, sticky=(N,S,E,W))

		lbox.bind('<<ListboxSelect>>', lambda x: self.remove_file_from_plot_area(lbox.curselection()))

		self.plots_manager.add(frame, text=str(len(self.plots_confs)+1))

		scalar_b = ttk.Button(frame, text='Scalar', command=lambda : self.plot(plot.Scalar), default='active')
		hist_b = ttk.Button(frame, text='Histogram', command=lambda : self.plot(plot.Histogram), default='active')
		image_b = ttk.Button(frame, text='Image', command=lambda : self.plot(plot.Image), default='active')
		scalar_b.grid(column=0, row=20, sticky=(S,W))
		hist_b.grid(column=0, row=20, sticky=S)
		image_b.grid(column=0, row=20, sticky=(S, E))

		exit_plot = ttk.Button(frame, text='close', command=lambda : self.delete_plot_area(frame), default='active')
		exit_plot.grid(column=0, row=22, sticky=S)

		self.plots_confs.append({
			'frame': frame,
			'lbox': lbox,
			'files': []
		})

	def plot(self, plot_class):
		idx_of_plots = self.plots_manager.index(self.plots_manager.select())
		files = [join(folder_, file_) for folder_, file_ in self.plots_confs[idx_of_plots]['files']]
		plot_class(files)

	def delete_plot_area(self, plotter_to_delete):
		if len(self.plots_confs) == 1:
			return
		idx = 0
		for plotter in self.plots_confs:
			if id(plotter['frame']) == id(plotter_to_delete):
				break
			idx += 1
		self.plots_manager.forget(idx)
		del self.plots_confs[idx]
		self.plots_manager.select(len(self.plots_confs)-1)

		for i in range(len(self.plots_confs)):
			self.plots_manager.tab(self.plots_confs[i]['frame'], text = str(i+1))

	def remove_file_from_plot_area(self, remove_index):
		if len(remove_index)==0:
			return
		remove_index = remove_index[0]
		idx_of_plots = self.plots_manager.index(self.plots_manager.select())

		del self.plots_confs[idx_of_plots]['files'][remove_index]

		self.plots_confs[idx_of_plots]['lbox'].delete(remove_index)

		for i in range(1,len(self.plots_confs[idx_of_plots]['files']),2):
			self.plots_confs[idx_of_plots]['lbox'].itemconfigure(i, background='#f0f0ff')
		for i in range(0,len(self.plots_confs[idx_of_plots]['files']),2):
			self.plots_confs[idx_of_plots]['lbox'].itemconfigure(i, background='#ffffff')


def main(args = None):
	if args is None:
		args = sys.argv[1:]

	args = [x for x in args if not isfile(x)]
	if len(args) == 0:
		args.append('.')

	root = WBNN_Plots()
	for folder in args:
		root.new_file_folder(folder)
	root.mainloop()

if __name__ == '__main__':
	main()