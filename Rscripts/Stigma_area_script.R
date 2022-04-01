# FROM OUTPUT OF ALGORITHM TO THERMAL TIME, FILTERING, PLOTTING AND GROWTH RATES FOR EACH GENOTYPE


library(openxlsx)
## A thermal time and data not yet prepared year 2019-20
# fixed samples
setwd ('U:/~~ TEMPORAL ~~/Field/stats-morphology/all_fixed_stigmas')

unfiltered_data <- read.table("data_all_fixed.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)

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


# 1. REMOVE OUTLIERS WITHIN THE SPIKES #

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
                                 stigma_area_mm=numeric(),
                                 stringsAsFactors=FALSE)

for (i in 1:v){
  
  subset_spike <- subset (unfiltered_data, new_id_1 == unique_ids[i], c("filename","new_id_1", "stigma", "stigma_area_mm"))
  out <- boxplot.stats(subset_spike$stigma_area_mm)$out # stigma_area_mm of outliers 
  out_ind <- which(subset_spike$stigma_area_mm %in% c(out)) # row number of those outliers
  a <- 1
  out_ind_length <- length(out_ind)
  outliers_spike_df1 <- data.frame(filename=character(),
                                   new_id_1=character(), 
                                   stigma=integer(),
                                   stigma_area_mm=numeric(),
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
check1 <- ddply(filtered_data_1, .(new_id_1), summarise, N = length(stigma_area_mm)) 
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
                                 stigma_area_mm=numeric(),
                                 stringsAsFactors=FALSE)

for (i in 1:v){
  
  subset_TP <- subset (filtered_data_1, new_id_2 == unique_ids[i], c("new_id_2", "plot", "spike", "stigma_area_mm"))
  out <- boxplot.stats(subset_TP$stigma_area_mm)$out
  out_ind <- which(subset_TP$stigma_area_mm %in% c(out))
  a <- 1
  out_ind_length <- length(out_ind)
  outliers_TP_df1 <- data.frame(new_id_2=character(), 
                                   plot=integer(),
                                   spike=integer(),
                                   stigma_area_mm=numeric(),
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

check2 <- ddply(filtered_data_2, .(new_id_2), summarise, N = length(stigma_area_mm)) 
min(check2$N)

nrow(unfiltered_data) #1702
nrow(filtered_data_1) #1588 - first round of filtering
nrow(filtered_data_2) #1474 - second round of filtering

## 3. FIND MAXIMUM AREA FOR EACH GENOTYPE AND THE CORRESPONDING THERMAL TIME AT WHICH IS REACHED

unique_ids <- unique(filtered_data_2$id)
v <- length(unique_ids)
i <- 1
plot_list <- list()
new.range <- seq(20, 450, by=4)
all_ids_fitted_values <- data.frame(id = as.character(),
                                    accumulated = as.numeric(),
                                    stigma_area = as.numeric(),
                                    percent_from_max = as.numeric())
library(ggplot2)

for (i in 1:v){
  
  subset_id <- subset(filtered_data_2, id == unique_ids[i])
  
  gp_i <- ggplot(subset_id, aes(x=accumulated, y=stigma_area_mm))+
    labs(x = "Cumulative degree days", y = "Stigma area (mm2)") +
    ggtitle(unique_ids[i])+
    stat_smooth(method = "loess", formula = y ~ x, span=0.9)+ # span can be modified to fit a more biologically reasonable curve
    geom_point()
  
  plot_list[[i]] <- gp_i
  gbuild <- ggplot_build(gp_i)
  
  exact_x_value_of_the_curve_maximum <- gbuild$data[[1]]$x[which((gbuild$data[[1]]$y) == max(gbuild$data[[1]]$y))]
  smooth_data <- loess(stigma_area_mm~accumulated, subset_id, span = 0.9)
  max.area <- predict(smooth_data, exact_x_value_of_the_curve_maximum)
  y_values <- predict(smooth_data, new.range)
  fitted_values <- data.frame(accumulated = as.numeric(new.range),
                              stigma_area = as.numeric(y_values))
  fitted_values$id <- as.character(unique_ids[i])
  
  fitted_values = ddply(fitted_values, .(accumulated), summarise, 
                        id = id,
                        accumulated = accumulated,
                        stigma_area = stigma_area,
                        percent_from_max = stigma_area*100/max.area)
  all_ids_fitted_values <- rbind(all_ids_fitted_values, fitted_values)
  fitted_values <- NULL
}


pdf()
plot_list
dev.off()

write.csv(unfiltered_data, "unfiltered_data.csv")
write.csv( filtered_data_2, "IQR_filtered_data.csv")
write.csv(all_ids_fitted_values, "loess_fitted_values.csv")

