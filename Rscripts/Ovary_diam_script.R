# FROM OUTPUT OF ALGORITHM TO THERMAL TIME, FILTERING, PLOTTING AND GROWTH RATES FOR EACH GENOTYPE


library(openxlsx)
## A thermal time and data not yet prepared year 2019-20
# fixed samples

setwd ("U:/~~ TEMPORAL ~~/Writing/Manuscripts/METHOD_PAPER/raw_data_ovary_2020_fixed.csv")

unfiltered_data <- read.table("U:/~~ TEMPORAL ~~/Writing/Manuscripts/METHOD_PAPER/raw_data_ovary_2020_fixed.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)

#separate filename column into different columns -> example: 24485_11DAH_1106_1-1_1.jpg
unfiltered_data$id <- as.character(lapply(strsplit(as.character(unfiltered_data$filename), "\\_"), "[", 1))
unfiltered_data$tp <- as.character(lapply(strsplit(as.character(unfiltered_data$filename), "\\_"), "[", 2))
unfiltered_data$date <- as.character(lapply(strsplit(as.character(unfiltered_data$filename), "\\_"), "[", 3))
unfiltered_data$rep <- as.character(lapply(strsplit(as.character(unfiltered_data$filename), "\\_"), "[", 4))
unfiltered_data$plot <- as.character(lapply(strsplit(as.character(unfiltered_data$rep), "\\-"), "[", 1))
unfiltered_data$spike <- as.character(lapply(strsplit(as.character(unfiltered_data$rep), "\\-"), "[", 2))
unfiltered_data$rep <- NULL

unfiltered_data$stigma <- as.character(lapply(strsplit(as.character(unfiltered_data$filename), "\\_"), "[", 5))
#keep the first number -> 1.jpg
unfiltered_data$stigma <- substr(unfiltered_data$stigma, 1, 1) 

# format date column

unfiltered_data$day <- as.character(substr(unfiltered_data$date, 1, 2))
unfiltered_data$month <- as.character(substr(unfiltered_data$date, 3, 4))
unfiltered_data$year <- '2020'
unfiltered_data$date <- paste0(unfiltered_data$day,"/",unfiltered_data$month,"/",unfiltered_data$year)
unfiltered_data$date <- as.Date(unfiltered_data$date, format="%d/%m/%y")
str(unfiltered_data)

# change GS59 to 0DAH in 2020 field data
unfiltered_data$tp <- gsub("GS59", "0DAH", unfiltered_data$tp)

# create heading date column. example: 24485_11DAH_1106_1-1_1.jpg -> heading (GS59) date = 11/06/2020 - 11 days = 31/05/2020

library(dplyr)
v <- length(unfiltered_data$id)
i <- 1

for (i in 1:v){
  
  unfiltered_data$days_to_GS59[i] <- gsub("[A-Z]", "", unfiltered_data$tp[i]) 
  unfiltered_data$heading <- as.Date(unfiltered_data$date) - as.numeric(unfiltered_data$days_to_GS59)
}

head(unfiltered_data)


#read temperature from data loggers
temp1 <- read.table("U:/~~ TEMPORAL ~~/Field/stats-morphology/temp_north.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)
temp2 <- read.table("U:/~~ TEMPORAL ~~/Field/stats-morphology/temp_south.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)

temp1_only <- temp1[,c("Time", "Date", "Celsius")] 
colnames(temp1_only)[3] <- "celsius1"
temp2_only <- temp2[,c("Time", "Date", "Celsius")] 
colnames(temp2_only)[3] <- "celsius2"
temp_all <- merge(temp1_only, temp2_only, by= c("Time", "Date"), all.x = TRUE)

#calculate averages of both readings 
temp_all$Tmean <- rowMeans(temp_all[3:4], na.rm=TRUE)

#calculate daily average  
library(plyr)
daily_mean = ddply(temp_all, .(Date), summarise, daily_temp = mean(Tmean))
daily_mean <- na.omit(daily_mean) 
daily_mean <-daily_mean[order(as.Date(daily_mean$Date, format="%d/%m/%y")), ]
daily_mean$Date <- as.Date(daily_mean$Date, format="%d/%m/%y")
str(daily_mean)


v <- length(unfiltered_data$id)
i <- 1

for (i in 1:v){
  
  row_i <- unfiltered_data[i,]
  temp_subset_i <- subset (daily_mean, Date >= as.Date(row_i$heading) & Date <= as.Date(row_i$date), select = c("Date", "daily_temp"))
  temp_subset_i <-temp_subset_i[order(as.Date(temp_subset_i$Date, format="%Y/%m/%d")), ]
  temp_subset_i$accumulated <-  cumsum(temp_subset_i$daily_temp)
  unfiltered_data$accumulated[i] <- last(temp_subset_i$accumulated)

}

head(unfiltered_data)

unfiltered_data <- read.table("U:/~~ TEMPORAL ~~/Writing/Manuscripts/METHOD_PAPER/unfiltered_data_ovary_2021_fixed.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)


#### 1. REMOVE OUTLIERS WITHIN THE SPIKES ####

# variation without filtering

# A. check variation within spikes
library(plyr)
variation_within = ddply(unfiltered_data, .(id, plot, spike, days_to_GS59, tp), summarise, 
                         ovary_diam_av = mean(ovary_diam_mm),
                         N_st = length(ovary_diam_mm),
                         sd_st = sd (ovary_diam_mm),
                         se_st = sd_st / sqrt(N_st),
                         accumulated_av = mean(accumulated),
                         CofV = (sd_st/ ovary_diam_av) * 100
)

head(variation_within)
summary(variation_within$CofV) 
# BEFORE FILTERING
# mean = 6.1906 (2020) & 5.634 (2021) vs 19.4245 (2021) for stigma area

mean(variation_within$sd_st, na.rm = TRUE) 
# BEFORE FILTERING
# mean = 0.1820579 (2020) & 0.1784803 (2021) vs 1.320841 (2021) for stigma area

hist(variation_within$CofV, main="Histogram of Coeffcient of variation (%) \nof ovary diam per spike \nIQR filtered data")

# B. check variation between spikes
variation_between = ddply(variation_within, .(id, days_to_GS59), summarise, 
                          ovary_diam_av_spikes = mean(ovary_diam_av),
                          N_st = length(ovary_diam_av),
                          sd_st = sd (ovary_diam_av),
                          se_st = sd_st / sqrt(N_st),
                          CofV = (sd_st/ ovary_diam_av_spikes) * 100
)

head(variation_between)
mean(variation_between$CofV, na.rm=TRUE) 
# BEFORE FILTERING = 5.827643 (2020) & 5.222164 (2021) vs 16.68374 (2021) for stigma area

mean(variation_between$sd_st, na.rm=TRUE) 
# BEFORE FILTERING = 0.1698833 (2020) & 0.1697686 (2021) vs 1.132542 (2021) for stigma area

hist(variation_between$CofV, main="Histogram of Coeffcient of variation (%) \nof ovary diam between spikes \nIQR filtered data")

## APPLY IQR CRITERIA

#create unique identifier for spike samples
unfiltered_data$new_id_1 <- paste(unfiltered_data$id, unfiltered_data$tp,unfiltered_data$plot,unfiltered_data$spike,sep='_')
head(unfiltered_data)

# for loop to identify outliers based on IQR criterion

library(dplyr)
unique_ids <- unique(unfiltered_data$new_id_1)
v <- length(unique_ids)
i <- 1


outliers_spike_df2 <- data.frame(filename=character(),
                                 new_id_1=character(), 
                                 stigma=integer(),
                                 ovary_diam_mm=numeric(),
                                 stringsAsFactors=FALSE)

for (i in 1:v){
  
  subset_spike <- subset (unfiltered_data, new_id_1 == unique_ids[i], c("filename","new_id_1", "stigma", "ovary_diam_mm"))
  out <- boxplot.stats(subset_spike$ovary_diam_mm)$out # ovary_diam_mm of outliers 
  out_ind <- which(subset_spike$ovary_diam_mm %in% c(out)) # row number of those outliers
  a <- 1
  out_ind_length <- length(out_ind)
  outliers_spike_df1 <- data.frame(filename=character(),
                                   new_id_1=character(), 
                                   stigma=integer(),
                                   ovary_diam_mm=numeric(),
                                   stringsAsFactors=FALSE)
    for (a in 1:out_ind_length){
      row_a <- subset_spike[out_ind[a],]
      outliers_spike_df1[a,] <- row_a
    }
  outliers_spike_df2 <- rbind(outliers_spike_df2, outliers_spike_df1)
}  

outliers_spike_df2 <- outliers_spike_df2[complete.cases(outliers_spike_df2), ]

# remove outliers from unfiltered data set

filtered_data_1<-unfiltered_data[!(as.character(unfiltered_data$filename) %in% as.character(outliers_spike_df2$filename)),]

# check the outliers have been removed correctly

any(filtered_data_1$filename %in% outliers_spike_df2$filename)

# make sure all spikes have at least measurements for 3 stigmas

library(plyr)
check1 <- ddply(filtered_data_1, .(new_id_1), summarise, N = length(ovary_diam_mm)) 
min(check1$N)  ## data 2019-20 because of pollen contamination in some cases I could only sample 1 or 2 stigmas. This is no due to the data filtering
               ## data 2020-21 is all ok                  

# 2. REMOVE OUTLIERS BETWEEN SPIKES FOR EACH TIME POINT #

#create unique identifier for spike samples
filtered_data_1$new_id_2 <- paste(filtered_data_1$id, filtered_data_1$tp, sep='_')
head(filtered_data_1)

# for loop to identify outliers based on IQR criterion

library(dplyr)
unique_ids <- unique(filtered_data_1$new_id_2)
v <- length(unique_ids)
i <- 1


outliers_TP_df2 <- data.frame(new_id_2=character(), 
                                 plot=integer(),
                                 spike=integer(),
                              ovary_diam_mm=numeric(),
                                 stringsAsFactors=FALSE)

for (i in 1:v){
  
  subset_TP <- subset (filtered_data_1, new_id_2 == unique_ids[i], c("new_id_2", "plot", "spike", "ovary_diam_mm"))
  out <- boxplot.stats(subset_TP$ovary_diam_mm)$out
  out_ind <- which(subset_TP$ovary_diam_mm %in% c(out))
  a <- 1
  out_ind_length <- length(out_ind)
  outliers_TP_df1 <- data.frame(new_id_2=character(), 
                                   plot=integer(),
                                   spike=integer(),
                                ovary_diam_mm=numeric(),
                                   stringsAsFactors=FALSE)
  for (a in 1:out_ind_length){
    row_a <- subset_TP[out_ind[a],]
    outliers_TP_df1[a,] <- row_a
  }
  outliers_TP_df2 <- rbind(outliers_TP_df2, outliers_TP_df1)
}  

outliers_TP_df2 <- outliers_TP_df2[complete.cases(outliers_TP_df2), ]
outliers_TP_df2$new_id_3 <- paste(outliers_TP_df2$new_id_2, outliers_TP_df2$plot, outliers_TP_df2$spike, sep='_')

# remove outliers from filtered_data_1

filtered_data_2<-filtered_data_1[!(as.character(filtered_data_1$new_id_1) %in% as.character(outliers_TP_df2$new_id_3)),]

# check the outliers have been removed correctly

any(filtered_data_2$new_id_1 %in% outliers_TP_df2$new_id_3)

# make sure all time points have at least measurements for 3 spikes

check2 <- ddply(filtered_data_2, .(new_id_2), summarise, N = length(ovary_diam_mm)) 
min(check2$N) # min = 3 (at least one spike is coming from each plot -> 2020)
              # min = 4

nrow(unfiltered_data) # (2020) = 2812 // (2021) = 1702 
nrow(filtered_data_1) # (2020) = 2651 // (2021) =  1561 // 1588 (area 2021) - first round of filtering
nrow(filtered_data_2) # (2020) = 2453 // (2021) = 1380 // 1474 (area 2021) - second round of filtering

# variation without filtering

# A. check variation within spikes
library(plyr)
variation_within = ddply(filtered_data_2, .(id, plot, spike, days_to_GS59, tp), summarise, 
                         ovary_diam_av = mean(ovary_diam_mm),
                         N_st = length(ovary_diam_mm),
                         sd_st = sd (ovary_diam_mm),
                         se_st = sd_st / sqrt(N_st),
                         accumulated_av = mean(accumulated),
                         CofV = (sd_st/ ovary_diam_av) * 100
)

head(variation_within)
summary(variation_within$CofV) 
# BEFORE FILTERING
# mean = 6.1906 (2020) & 5.634 (2021) vs 19.4245 (2021) for stigma area
# AFTER FILTERING
# mean = 5.1069 (2020) & 4.074 (2021) vs 16.1802 (2021) for stigma area

mean(variation_within$sd_st, na.rm = TRUE) 
# BEFORE FILTERING
# mean = 0.1820579 (2020) & 0.1784803 (2021) vs 1.320841 (2021) for stigma area
# AFTER FILTERING
# mean = 0.1488599 (2020) & 0.1290937 (2021) vs 1.102552 (2021) for stigma area

hist(variation_within$CofV, main="Histogram of Coeffcient of variation (%) \nof ovary diam per spike \nIQR filtered data")

# B. check variation between spikes
variation_between = ddply(variation_within, .(id, days_to_GS59), summarise, 
                          ovary_diam_av_spikes = mean(ovary_diam_av),
                          N_st = length(ovary_diam_av),
                          sd_st = sd (ovary_diam_av),
                          se_st = sd_st / sqrt(N_st),
                          CofV = (sd_st/ ovary_diam_av_spikes) * 100
)

head(variation_between)
mean(variation_between$CofV, na.rm=TRUE) 
# BEFORE FILTERING = 5.827643 (2020) & 5.222164 (2021) vs 16.68374 (2021) for stigma area
# AFTER FILTERING = 5.47071 (2020) & 4.530405 (2021) vs 17.3062 (2021) for stigma area


mean(variation_between$sd_st, na.rm=TRUE) 
# BEFORE FILTERING = 0.1698833 (2020) & 0.1697686 (2021) vs 1.132542 (2021) for stigma area
# AFTER FILTERING = 0.1611308 (2020) & 0.1458822 (2021) vs 1.181685 (2021) for stigma area

hist(variation_between$CofV, main="Histogram of Coeffcient of variation (%) \nof ovary diam between spikes \nIQR filtered data")


## 3. FIND MAXIMUM DIAM FOR EACH GENOTYPE AND THE CORRESPONDING THERMAL TIME AT WHICH IS REACHED

unique_ids <- unique(filtered_data_2$id)
v <- length(unique_ids)
i <- 1
plot_list <- list()
new.range <- seq(20, 450, by=1)
all_ids_fitted_values <- data.frame(id = as.character(),
                                    accumulated = as.numeric(),
                                    ovary_diam = as.numeric(),
                                    percent_from_max = as.numeric())
library(ggplot2)

for (i in 1:v){
  
  subset_id <- subset(filtered_data_2, id == unique_ids[i])
  
  gp_i <- ggplot(subset_id, aes(x=accumulated, y=ovary_diam_mm))+
    labs(x = "Cumulative degree days", y = "Ovary diameter (mm)") +
    ggtitle(unique_ids[i])+
    stat_smooth(method = "loess", formula = y ~ x, span=0.9)+ # span can be modified to fit a more biologically reasonable curve
    geom_point()
  
  plot_list[[i]] <- gp_i
  gbuild <- ggplot_build(gp_i)
  
  exact_x_value_of_the_curve_maximum <- gbuild$data[[1]]$x[which((gbuild$data[[1]]$y) == max(gbuild$data[[1]]$y))]
  smooth_data <- loess(ovary_diam_mm~accumulated, subset_id, span = 0.9)
  max.diam <- predict(smooth_data, exact_x_value_of_the_curve_maximum)
  y_values <- predict(smooth_data, new.range)
  fitted_values <- data.frame(accumulated = as.numeric(new.range),
                              ovary_diam = as.numeric(y_values))
  fitted_values$id <- as.character(unique_ids[i])
  
  fitted_values = ddply(fitted_values, .(accumulated), summarise, 
                        id = id,
                        accumulated = accumulated,
                        ovary_diam = ovary_diam,
                        percent_from_max = ovary_diam*100/max.diam)
  all_ids_fitted_values <- rbind(all_ids_fitted_values, fitted_values)
  fitted_values <- NULL
}


pdf()
plot_list
dev.off()

write.csv(unfiltered_data, "U:/~~ TEMPORAL ~~/Writing/Manuscripts/METHOD_PAPER/unfiltered_data_ovary_2021_fixed.csv")
write.csv( filtered_data_2, "U:/~~ TEMPORAL ~~/Writing/Manuscripts/METHOD_PAPER/IQR_filtered_data_ovary_2021_fixed.csv")
write.csv(all_ids_fitted_values, "U:/~~ TEMPORAL ~~/Writing/Manuscripts/METHOD_PAPER/loess_fitted_values_ovary_2021_fixed.csv")

