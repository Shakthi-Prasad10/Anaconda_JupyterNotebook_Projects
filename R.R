library(dplyr)
library(ggplot2)

annual <- df %>%
group_by(year) %>%
summarise(
total_co2 = sum(co2, na.rm= TRUE),
mean_co2= mean(co2, na.rm= TRUE),
.groups = "drop"
)

ggplot(annual, aes(year, total_co2))+
geom_line()+
labs(title = "Global total CO2 emissions",x= "year", y = "Mt CO2")