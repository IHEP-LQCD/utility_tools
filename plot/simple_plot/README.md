## Simple plot
---
`plotm.py` is a command line plotting script based on Python.
For convience, you can add this script to your `PATH` environment.
Type `plotm.py -h` to see the usage of script, 
```
usage: plotm.py [-h] [-i INFILE [INFILE ...]] [-o OUTFILE] [-xl XLOW]
                [-xh XHIGH] [-yl YLOW] [-yh YHIGH] [-s SCALE [SCALE ...]]
                [-x XLABEL] [-y YLABEL] [-l] [-b] [-c COL [COL ...]]
                [-tkxy TKXY TKXY TKXY TKXY] [-hline HLINE [HLINE ...]]
                [-errband ERRBAND [ERRBAND ...]]

optional arguments:
  -h, --help            show this help message and exit
  -i INFILE [INFILE ...], --infile INFILE [INFILE ...]
  -o OUTFILE, --outfile OUTFILE
  -xl XLOW, --xlow XLOW
  -xh XHIGH, --xhigh XHIGH
  -yl YLOW, --ylow YLOW
  -yh YHIGH, --yhigh YHIGH
  -s SCALE [SCALE ...], --scale SCALE [SCALE ...]
  -x XLABEL, --xlabel XLABEL
                        x-axis label of the plot (default: $t/a_t$)
  -y YLABEL, --ylabel YLABEL
                        y-axis label of the plot (default: $m_{eff}(t)$)
  -l, --logy            whether to use log scale in y-axis (default: False)
  -b, --binary          whether the file is binary or not (default: False)
  -c COL [COL ...], --col COL [COL ...]
                        Columns need to draw, counting from 0 (default: None)
  -tkxy TKXY TKXY TKXY TKXY, --tkxy TKXY TKXY TKXY TKXY
                        Major and minor ticks of X and Y axis (default: [5.0,
                        1.0, 1.0, 0.1])
  -hline HLINE [HLINE ...], --hline HLINE [HLINE ...]
                        Plot a horizontal line (default: None)
  -errband ERRBAND [ERRBAND ...], --errband ERRBAND [ERRBAND ...]
                        Plot the error band: mean, err, xmin, ymin (default:
                        None)
```
where the most frequently used options are:
* `-i`, specify the input filename of data, can be single or multiple files
* `-o`, the output name of the plotted figure, usually one should use high quality **pdf** format, here for convenience of embedding the figure into the markdown file, we can use **png** format instead, to change the output figure format, just the line of `plt.savefig(outfile+".png")` to the format you like, e.g. `plt.savefig(outfile+".pdf")` for pdf format
* `-xl`, the lower limit of x-axis 
* `-xh`, the upper limit of x-axis
* `-yl`, the lower limit of y-axis
* `-yh`, the upper limit of y-axis
* `-s`, scale the data with some common factor
* `-x`, the x-axis label, default is $t/a_t$
* `-y`, the y-axis label, default is $m_{eff}(t)$
* `-l`, whether to log scale in y-axis, the default is no
* `-b`, whether the input file is in binary format, default if no
* `-c`, choose the columns of data to plot, for example `-c 0 1` will plot with 0th-column as x-axis data and 1th-column as y-axis data, `-c 0 3 4` will plot the errorbar plot with 0th-column as x-axis data, 3th-column as y-axis mean, 4th-column as y-axis error
* `-tkxy`, specify the ticks parameter of x and y-axis, e.g. `-tkxy 5 1 0.5 0.1` will plot the x-axis with major ticks 5, minor ticks 1, and y-axis with major ticks 0.5, minor ticks 0.1
* `-hline`, plot a horizontal line at a specific y-axis value
* `-errband`, plot an error band along with specified mean, error, lower and upper limit of x-axis

You can add whatever options suitable fairly easy by modify this script.
