library(ggplot2)
library(glue)
library(RcppCNPy)
source("https://gist.githubusercontent.com/benmarwick/2a1bb0133ff568cbe28d/raw/fb53bd97121f7f9ce947837ef1a4c65a73bffb3f/geom_flat_violin.R")

mypath = "/Users/myfiles/Documents/EPFL/M_I/MLE/scRNAseq/results/"

# Creates raincloud plots
raincloud_plotter = function(pred_f,     # Numpy file containing predictions
                             celltype_f, # csv containing celltype names
                             savefile,   # Name of output file
                             path_app=T, # Append path to mypath (T or F)
                             silent=TRUE,# Print output plot
                             title=NULL, # Optional title to add to plot
                             dims = NULL,# Dimensions of output pdf,
                                         # vector with element (width, height)
                             col = 2     # Column of npy array containing probability
                                         # disregarded if 1D array
                             ) {
  if (path_app) {
    pred_f = paste(mypath, pred_f, sep="")
    celltype_f = paste(mypath, celltype_f, sep="")
    savefile = paste(mypath, savefile, sep="")
  }

  
  pred = npyLoad(pred_f)
  
  if (!is.null(dim(pred))) {
    pred = pred[,col]
  }
  celltype = read.csv(celltype_f)[,2]
  
  raincloud_theme = theme(
    text = element_text(size = 10),
    axis.title.x = element_text(size = 16),
    axis.title.y = element_blank(),
    axis.text = element_text(size = 11),
    axis.text.x = element_text(angle = 45, vjust = 0.5),
    legend.title=element_text(size=16),
    legend.text=element_text(size=16),
    legend.position = "right",
    plot.title = element_text(lineheight=.8, face="bold", size = 16),
    panel.border = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.line.x = element_line(colour = 'black', size=0.5, linetype='solid'),
    axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'))
  
  tmp = data.frame(value = pred, variable = as.factor(celltype))
  g <- ggplot(data = tmp, aes(y = value, x = variable, fill = variable)) +
    geom_flat_violin(position = position_nudge(x = .2, y = 0), alpha = .6) +
    geom_point(aes(y = value, color = variable),
               position = position_jitter(width = .15), size = .5, alpha = .6) +
    geom_boxplot(width = .1, outlier.shape = NA, alpha = 0.3) +
    expand_limits(x = 5.25) +
    coord_flip(ylim=c(0, 1)) +
    theme_bw() +
    guides(fill = FALSE) +
    guides(color = FALSE) +
    ylab("Probability") +
    raincloud_theme 
  if (!is.null(title)) g = g + ggtitle(title)
  if (!silent) print(g)
  if (!is.null(savefile)) {
    if (is.null(dims)) {
      w = 8.27
      h = 11.69
    } else {
      w = dims[1]
      h = dims[2]
    }
    ggsave(filename = savefile,
           plot=g, device = pdf(), useDingbats=FALSE, width = w, height = h)
    dev.off()
  }
  return(g)
}
