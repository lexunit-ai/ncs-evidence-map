###============================================
### Overview:
### Replication code to create the country-level
### analysis figures in the main text and SI
### for the manuscript, "A global evidence map of human well-being 
### and biodiversity co-benefits and trade-offs of natural climate solutions"
###============================================

###============================================
### Load packages
###============================================
library(ggplot2)
library(ggrepel)
library(dplyr)
library(stringr)
library(cowplot)
library(pals)
library(biscale)
# library(rnaturalearth)

###============================================
### Summarizing evidence and covariates
### by country using the TNC NCS Atlas
###============================================

### Country-level data
countrySum <- readr::read_csv("data/countryLevelData.csv")

### Global map
rneworld <- sf::st_as_sf(rnaturalearth::countries110) # use ISO_A3
  # Remove Antarctica
rneworld <- rneworld %>%
  dplyr::filter(NAME != "Antarctica")
  # Cleanup for missing ISO codes
rneworld$ISO_A3[which(rneworld$NAME=="France")] <- "FRA"
rneworld$ISO_A3[which(rneworld$NAME=="Norway")] <- "NOR"
  # Use robinson projection
robinson = "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
rnewproj <- sf::st_transform(rneworld, robinson)

### Constructing spatial dataset
countryGeo <- countrySum %>%
  left_join(rnewproj[,c("ISO_A3","POP_EST","POP_YEAR","NAME","geometry")], by=c("AlphaISO"="ISO_A3")) %>%
  mutate(POP_EST = as.numeric(POP_EST))

### Find thresholds
manageCthreshold <- median(countrySum$manageC)
protectCthreshold <- median(countrySum$protectC)
restoreCthreshold <- median(countrySum$restoreC)
manageEvidenceT <- median(countrySum$manageEvidence)
protectEvidenceT <- median(countrySum$protectEvidence)
restoreEvidenceT <- median(countrySum$restoreEvidence)

### Create proportions for threatened SR and remaining HDI for plotting
countrySum$SRp <- countrySum$SR/max(countrySum$SR)
countrySum$HDIremain <- 1-countrySum$HDI
countrySum$HDIp <- countrySum$HDIremain/max(countrySum$HDIremain)

###============================================
### FIGURES
###============================================

### Half circle points
shapeLeft="\u25D6"
shapeRight="\u25D7"
sr_color = "#377eb8"
dev_color = "#e66101"

### Color for boundary lines in map
country_boundary_color <- "gray20" 

### Use biscale to generate color ramp
custom_pal <- bi_pal(pal="BlueYl",dim=4,flip_axes=TRUE,rotate_pal=FALSE,preview=FALSE)
custom_pal[c(1:8)] <- custom_pal[1]

###*****************************
### PROTECT: Map
###*****************************

### Construct your data
bi_data = biscale::bi_class(countryGeo, x=protectEvidence, y=protectC, style="quantile", dim=4)

### Tack on new protect column for output
countrySum <- countrySum %>%
  mutate(protectEvidenceCat = unlist(str_split(bi_data$bi_class,"-"))[seq(1,362,by=2)],
         protectCcat = unlist(str_split(bi_data$bi_class,"-"))[seq(2,362,by=2)])

### Map for protect C
protectCmap <- ggplot(
  data = bi_data
) +
  scale_alpha(name = "",
              range = c(0.6, 0),
              guide = F) + # suppress legend
  geom_sf( # base map
    data=rneworld, aes(geometry=geometry, fill="white"),
    color=country_boundary_color,size=0.1, inherit.aes = FALSE
  ) + 
  geom_sf(
    aes( geometry = geometry,
         fill = bi_class
    ),
    color = country_boundary_color,
    size = 0.1, show.legend=FALSE
  ) +
  bi_scale_fill(pal=custom_pal, dim=4) +
  coord_sf(xlim=c(-13210131,15810131),ylim=c(-5900074,8343004), expand=FALSE) +
  # add titles
  labs(x = NULL,y = NULL) + guides(fill="none") +
  theme_void()

# View figure
protectCmap

###*****************************
### PROTECT: Scatterplot
###*****************************
p_protect <- ggplot(countrySum, aes(x=protectEvidence,y=protectC,label=AlphaISO))
p_protect <- p_protect + 
  geom_text(aes(size=SRp), label=shapeLeft,alpha=0.7, color=sr_color,family = "Arial Unicode MS") +
  geom_text(aes(size=HDIp), label=shapeRight, alpha=0.7, color=dev_color,family = "Arial Unicode MS")
p_protect <- p_protect + scale_radius(range=c(5,12))
p_protect <- p_protect + geom_vline(xintercept=protectEvidenceT, linetype=2) + geom_hline(yintercept = protectCthreshold,linetype=2)
p_protect <- p_protect + geom_text_repel(segment.color="black",
                                         size=2,
                                         force=5,
                                         max.overlaps=10,
                                         force_pull=0.5,
                                         max.iter=20000)
p_protect <- p_protect + scale_y_log10()
p_protect <- p_protect + scale_x_log10()
p_protect <- p_protect + theme_classic()
p_protect <- p_protect + labs(x="Evidence",y="Protect C")
p_protect <- p_protect + guides(size=guide_legend(title="Proportion:",nrow=1)) + 
  theme(legend.position="top",
        text = element_text(family = "Arial Unicode MS"),
        legend.text = element_text(size=7),
        legend.title = element_text(size=8),
        legend.spacing.x = unit(0, 'mm'))

# View figure
p_protect

###*****************************
### MANAGE: Map
###*****************************

### Construct your data
bi_data = biscale::bi_class(countryGeo, x=manageEvidence, y=manageC, style="quantile", dim=4)

### Tack on new manage column for output
countrySum <- countrySum %>%
  mutate(manageEvidenceCat = unlist(str_split(bi_data$bi_class,"-"))[seq(1,362,by=2)],
         manageCcat = unlist(str_split(bi_data$bi_class,"-"))[seq(2,362,by=2)])

### Map for manage C
manageCmap <- ggplot(
  # use the same dataset as before
  data = bi_data
) +
  scale_alpha(name = "",
              range = c(0.6, 0),
              guide = F) + # suppress legend
  geom_sf( # base map
    data=rneworld, aes(geometry=geometry, fill="white"),
    color=country_boundary_color,size=0.1, inherit.aes = FALSE
  ) + 
  geom_sf(
    aes( geometry = geometry,
         fill = bi_class
    ),
    color = country_boundary_color,
    size = 0.1, show.legend=FALSE
  ) +
  bi_scale_fill(pal=custom_pal, dim=4) +
  coord_sf(xlim=c(-13210131,15810131),ylim=c(-5900074,8343004), expand=FALSE) +
  # add titles
  labs(x = NULL,y = NULL) + guides(fill="none") +
  theme_void()

###*****************************
### MANAGE: Scatterplot
###*****************************

p_manage <- ggplot(countrySum, aes(x=manageEvidence,y=manageC,label=AlphaISO))
p_manage <- p_manage + 
  geom_text(aes(size=SRp), label=shapeLeft,alpha=0.7, color=sr_color,family = "Arial Unicode MS") +
  geom_text(aes(size=HDIp), label=shapeRight, alpha=0.7, color=dev_color,family = "Arial Unicode MS")
p_manage <- p_manage + scale_radius(range=c(5,12))
p_manage <- p_manage + geom_vline(xintercept=manageEvidenceT, linetype=2) + geom_hline(yintercept = manageCthreshold,linetype=2)
p_manage <- p_manage + geom_text_repel(segment.color="black",
                                       size=2,
                                       force=5,
                                       max.overlaps=10,
                                       force_pull=0.5,
                                       max.iter=20000)
p_manage <- p_manage + scale_y_log10()
p_manage <- p_manage + scale_x_log10(breaks=c(1,10,100,1000,10000),labels=c(1,10,100,1000,10000), limits=c(1,50000))
p_manage <- p_manage + theme_classic()
p_manage <- p_manage + labs(x="",y="Manage C")
p_manage <- p_manage + theme(legend.position="none",text = element_text(family = "Arial Unicode MS"))

###*****************************
### RESTORE: Map
###*****************************

### Construct your data
bi_data = biscale::bi_class(countryGeo, x=restoreEvidence, y=restoreC, style="quantile", dim=4)

### Tack on new restore column for output
countrySum <- countrySum %>%
  mutate(restoreEvidenceCat = unlist(str_split(bi_data$bi_class,"-"))[seq(1,362,by=2)],
         restoreCcat = unlist(str_split(bi_data$bi_class,"-"))[seq(2,362,by=2)])

### Map for restore C
restoreCmap <- ggplot(
  # use the same dataset as before
  data = bi_data
) +
  scale_alpha(name = "",
              range = c(0.6, 0),
              guide = F) + # suppress legend
  geom_sf( # base map
    data=rneworld, aes(geometry=geometry, fill="white"),
    color=country_boundary_color,size=0.1, inherit.aes = FALSE
  ) + 
  geom_sf(
    aes( geometry = geometry,
         fill = bi_class
    ),
    color = country_boundary_color,
    size = 0.1, show.legend=FALSE
  ) +
  bi_scale_fill(pal=custom_pal, dim=4) +
  coord_sf(xlim=c(-13210131,15810131),ylim=c(-5900074,8343004), expand=FALSE) +
  # add titles
  labs(x = NULL,y = NULL) + guides(fill="none") +
  theme_void()

###*****************************
### RESTORE: Scatterplot
###*****************************
p_restore <- ggplot(countrySum, aes(x=restoreEvidence+1,y=restoreC,label=AlphaISO))
p_restore <- p_restore + 
  geom_text(aes(size=SRp), label=shapeLeft,alpha=0.7, color=sr_color,family = "Arial Unicode MS") +
  geom_text(aes(size=HDIp), label=shapeRight, alpha=0.7, color=dev_color,family = "Arial Unicode MS")
p_restore <- p_restore + scale_radius(range=c(5,12))
p_restore <- p_restore + geom_vline(xintercept=restoreEvidenceT, linetype=2) + geom_hline(yintercept = restoreCthreshold,linetype=2)
p_restore <- p_restore + geom_text_repel(segment.color="black",
                                         size=2,
                                         force=5,
                                         max.overlaps=10,
                                         force_pull=0.5,
                                         max.iter=20000)
p_restore <- p_restore + scale_y_log10()
p_restore <- p_restore + scale_x_log10()
p_restore <- p_restore + theme_classic()
p_restore <- p_restore + labs(x="Evidence",y="Restore C")
p_restore <- p_restore + theme(legend.position="none",text = element_text(family = "Arial Unicode MS"))
#p_restore
