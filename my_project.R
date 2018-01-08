install.packages(c("data.table","ggplot2","gridExtra","reshape","scales"), repos="http://cran.us.r-project.org")
install.packages(c("GGally"))
install.packages("stR")
install.packages(c("maps", "mapdata"))
install.packages(c("ggplot2", "devtools", "dplyr", "stringr"))
install.packages("devtools")
install.packages("psych", dependencies = TRUE)
install.packages("GPArotation")

library(ggplot2)
library(ggthemes)
library(dplyr)
library(gridExtra)
library(tidyr)
library(MASS)
library(scales)
library(GGally)
library(psych)
library(GPArotation)



# This report explores dataset containing information about xxx number loans 
# There is 81 variables with 113937 observations

# let's load my data:

```{r echo=FALSE, Load_the_Data}
# Load the Data
getwd()
setwd('/Users/tommyly/Documents/Udacity/p4 - Explore and Summarize Data with R/data')
# Read the csv file

getwd()
setwd("C:/Users/Marcela/Desktop/Udacity/R/project/4")

prosloan = read.csv("prosperLoanData.csv")

omega(prosloan)
sessionInfo()

set.seed(2000)
subject= 100
experiences= 200
trials= 2
maxIter= 200
convergLimit= .000001
dim(prosloan)
# We should see some basic statistics for our vairables:
describe(prosloan)
warnings(prosloan)
error.bars(prosloan)
r <- lowerCor(prosloan)
alpha(prosloan)
graphics.off()
par("mar")
par(mar=c(1,1,1,1))
pch='.')
pairs.panels(prosloan, pch='.')
outlier(prosloan)


# to powinno byc na dole w console
str(prosloan)
names(prosloan)
str(prosloan)
nrow(prosloan)
ncol(prosloan)


prosloan$ProsperRating.alpha = factor(prosloan$ProsperRating..Alpha.,
                                     levels = c("AA","A","B","C","D","E","HR","NA"))

prosloan$ProsperScore = factor(prosloan$ProsperScore)

# to nie dzilaa przez alpha
ggplot(data = na.omit(prosloan), aes(ProsperRating.alpha)) +
  geom_histogram(aes(fill = ProsperRating.alpha)) +
  ggtitle('Numbers of Loans by Prosper Rating') +
  xlab('Rating') +
  ylab('Number of Loans')
summary(prosloan$ProsperRating.alpha)

# to jest ok 
summary(prosloan$IncomeRange)
# to nie dziala nie wiem czemu ???????????
ggplot(data = prosloan, x = Occupation, y = IncomeRange) + 
  geom_histogram(aes(fill = ProsperRating.alpha))

#to jest ok pokazuje co jest w stanach
summary(prosloan$BorrowerState)
# In California people take the biggest amounts of loans!


# z tym jest coœ nie tak wow to jest niesamowite
```{r echo=FALSE, Load_the_Data}
ggplot(data = na.omit(prosloan), aes(ProsperRating)) +
  geom_bar(aes(fill = ProsperRating)) +
  ggtitle('Numbers of Loans by Prosper Rating') +
  xlab('Rating') +
  ylab('Number of Loans')
```
### Surprisigny it's not the richest ppl that take most loans
# As we can see people with Prosper Rating "E" are the most common borrowers 
summary(prosloan$ProsperRating.alpha)


# 1 nie jestem pewna czy jest to wiarygodne ale moze tak to z LV
ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) +
  geom_bar(color="black", fill = '#007EE5') +
  ggtitle('Is Borrower Home Owner') +
  xlab('Is Borrower Home Owner') +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) +
  ylab('Count')
  scale_fill_brewer(palette="Dark2")  
# to jest te¿ spoko
summary(prosloan$IsBorrowerHomeowner)

#tu jest cos nie tak
ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) + 
  geom_bar(color="black", fill = ) + 
  ggtitle('Is Borrower Home Owner') + 
  xlab('Is Borrower Home Owner') + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) + 
  ylab('Count') + 
  scale_fill_manual(values=c("#999999", "#E69F00", "#999999"))

#scale_fill_brewer(palette="Dark2")  
  

# scale_fill_brewer(palette="Dark2")

######2 Loan Status juz jest
ggplot(data = prosloan, aes(x = LoanStatus)) +
  geom_bar(color="black", fill = '#007EE5') +
  ggtitle('Loan Status') +
  xlab('Loan Status') +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) +
  ylab('Count')

# From all our loans a big count is already completed, 
# quite many was cancelled aswell

summary(prosloan$LoanStatus)
##3. Income Range to od tego goscia chyba
set.seed(1234)
x <- rnorm(200)
# Histogram
hp<-qplot(x =x, fill=..count.., geom="histogram") 
hp
# Sequential color scheme
hp+scale_fill_gradient(low="blue", high="red")

# 4 People earining 25.000 - 49.999$ tend to borrow more money

ggplot(data = prosloan, aes(IncomeRange)) +
  geom_bar(color="black", fill = '#007EE5') +
  ggtitle('Borrower Income Range') +
  xlab('Income') +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) +
  ylab('Count')



  
# Loan original amount to jest chyba srednie ale zostawie
ggplot(data = prosloan, aes(LoanOriginalAmount)) +
  geom_bar(color="black", fill = '#007EE5') +
  ggtitle('Loan Original Amount') +
  xlab('Loan Amounts') +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) +
  ylab('Count') 


## 5 do zminay tez zeby bylo inne
ggplot(data = prosloan, aes(x = ProsperScore)) + 
  geom_bar(color = "black", fill = '#007EE5') +
  theme(axis.text.x = element_text(angle = 60, vjust = 0.6)) +
  xlab("Score") + ggtitle("Prosper Score Distribution")
summary(prosloan$ProsperScore)

## 6 to moj pomysl ale chyba nie ma sensu bo nie sa podzielone miesaice to nie ma sensu
ggplot(data = prosloan, aes(x = MonthlyLoanPayment)) + 
  geom_bar(color = "black", fill = '#007EE5') +
  theme(axis.text.x = element_text(angle = 40, vjust = 0.4)) +
  xlab("Score") + ggtitle("MonthlyLoanPayment")
summary(prosloan$MonthlyLoanPayment)


# 7Number of Loans  by Month od tego kolesia to tutaj LV

prosloan$LoanOriginationDate.month=format(prosloan$LoanOriginationDate, "(%m) %b")
months <- c('1', '2', '3', '4', '5', '6',
            '7', '8', '9', '10', '11', '12')

prosloan$LoanOriginationDate.month=format(prosloan$LoanOriginationDate, "(%m) %b")

Sys.setlocale("LC_TIME", "C");

set.seed(1234)
x <- rnorm(200)
# Histogram
hp<-qplot(x =x, fill=..count.., geom="histogram") 
hp
ggplot(prosloan, aes(LoanOriginationDate.month)) +
  geom_histogram(stat="Count", color = 'black', fill = '#007EE5') +
  facet_wrap(~LoanOriginationDate.year) +
  ggtitle('Number of Loans by Month') +
  xlab('Month') +
  ylab('Number of Loans') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# 8 to jest tez od tego goscia albo moze moje
# The most popular income is between 4-5k$
ggplot(aes(x = StatedMonthlyIncome), data = prosloan) +
  geom_histogram(binwidth = 1000) +
  scale_x_continuous(
    limits = c(0, quantile(prosloan$StatedMonthlyIncome, 0.99,
                           na.rm = TRUE)),
    breaks = seq(0, quantile(prosloan$StatedMonthlyIncome, 0.99, 
                             na.rm = TRUE), 1000)) +
  theme(axis.text.x = element_text(angle = 90))



########probouje jeszcze raz w koncu jest i daje na pierwszy graf
# Smaller amounts seems to be more used
ggplot(aes(x = LoanOriginalAmount), data = prosloan) +
  geom_histogram(binwidth = 1000) +
  scale_x_continuous(
    limits = c(0, quantile(prosloan$StatedMonthlyIncome, 0.99,
                           na.rm = TRUE)),
    breaks = seq(0, quantile(prosloan$StatedMonthlyIncome, 0.99, 
                             na.rm = TRUE), 1000)) +
  theme(axis.text.x = element_text(angle = 90))


# 10 ok ale siê wolno ³aduje nawet bardzo wolno
ggplot(data = prosloan, aes(x = LoanOriginalAmount, y = MonthlyLoanPayment)) +
  geom_point() +
  scale_x_discrete(breaks = seq(0, 203, 11)) +


#### 11 to jest kopia 3 ale chyba moze zostac
ggplot(data = prosloan, aes(x = StatedMonthlyIncome, y = LoanOriginalAmount)) +
  geom_point() +
  scale_x_discrete(breaks = seq(0, 203, 11))

# to jest kopia ale nie ma sensu, musi byc inny grafik
ggplot(data = prosloan, aes(x = Occupation, y = IncomeRange)) +
  geom_point() +
  scale_x_discrete(breaks = seq(0, 203, 11))

#12 heatmap to jest zajebiste!!! jestem z siebie dumna wow

#13 to tylko na probe:
ggplot(aes( x = EmploymentStatus,y = IncomeRange, fill = LoanOriginalAmount),
       data = prosloan) +
  geom_tile() +
  scale_fill_gradientn(colours = colorRampPalette(c('purple', 'yellow'))(100))

#14 to jest fantstyczne tylko inne kolory
ggplot(aes( x = CreditGrade, y = IncomeRange, fill = LoanOriginalAmount),
       data = prosloan) +
  geom_tile() +
  scale_fill_gradientn(colours = colorRampPalette(c('blue', 'yellow'))(100))
       
# dziwne ze to nie dziala
prosloan$LoanOriginationDate <- as.POSIXct(prosloan$LoanOriginationDate,
                                          format="%Y-%m-%d")

prosloan$LoanOriginationDate.year <-prosloan$LoanOriginationDate %>% 
  format('%Y') %>% strtoi()

# to nie dziala 
ggplot(prosloan, aes(as.factor(LoanOriginationDate.year))) +
  geom_histogram(color = 'black', fill = '#007EE5') +
  ggtitle('Number of Loans by Year') +
  stat_bin(geom="text", 
           aes(label=..count.., vjust=-0.9, hjust=0.5)) +
  xlab('Year') +
  ylab('Number of Loans')
summary(prosloan$LoanOriginationDate.year)



Prosper Data - BorrowerRate - Prosper Rating
prosloan$ProsperRating.alpha <- factor(prosloan$ProsperRating.alpha)

ggplot(data = prosloan, aes(x = Occupation, y = LoanOriginalAmount)) +
  geom_boxplot() +
  xlab("Prosper Rating") +
  ggtitle("Borrower Rate for Different Prosper Rating")



devtools::install_github("dkahle/ggmap")

library(ggplot2)
library(ggmap)
library(maps)
library(mapdata)
library(maps)
library(ggmap)

us<- map_data("state")
usa <- ggplot2::map_data("state")


dim(usa)
#> [1] 7243    6

head(usa)
#>        long      lat group order region subregion
#> 1 -101.4078 29.74224     1     1   main      <NA>
#> 2 -101.3906 29.74224     1     2   main      <NA>
#> 3 -101.3620 29.65056     1     3   main      <NA>
#> 4 -101.3505 29.63911     1     4   main      <NA>
#> 5 -101.3219 29.63338     1     5   main      <NA>
#> 6 -101.3047 29.64484     1     6   main      <NA>

tail(usa)
#>           long      lat group order         region subregion
#> 7247 -122.6187 48.37482    10  7247 whidbey island      <NA>
#> 7248 -122.6359 48.35764    10  7248 whidbey island      <NA>
#> 7249 -122.6703 48.31180    10  7249 whidbey island      <NA>
#> 7250 -122.7218 48.23732    10  7250 whidbey island      <NA>
#> 7251 -122.7104 48.21440    10  7251 whidbey island      <NA>
#> 7252 -122.6703 48.17429    10  7252 whidbey island      <NA>
#Here is the high-res world map centered on the Pacific Ocean from mapdata


us <- ggplot2::map_data("state")

states <- ggplot2::map_data("state")
dim(states)
#> [1] 15537     6

head(states)
#>        long      lat group order  region subregion
#> 1 -87.46201 30.38968     1     1 alabama      <NA>
#> 2 -87.48493 30.37249     1     2 alabama      <NA>
#> 3 -87.52503 30.37249     1     3 alabama      <NA>
#> 4 -87.53076 30.33239     1     4 alabama      <NA>
#> 5 -87.57087 30.32665     1     5 alabama      <NA>
#> 6 -87.58806 30.32665     1     6 alabama      <NA>

tail(states)
#>            long      lat group order  region subregion
#> 15594 -106.3295 41.00659    63 15594 wyoming      <NA>
#> 15595 -106.8566 41.01232    63 15595 wyoming      <NA>
#> 15596 -107.3093 41.01805    63 15596 wyoming      <NA>
#> 15597 -107.9223 41.01805    63 15597 wyoming      <NA>
#> 15598 -109.0568 40.98940    63 15598 wyoming      <NA>
#> 15599 -109.0511 40.99513    63 15599 wyoming      <NA>

# to znowu coœ nowego zeby zrobic mapke ta jest najlepsza ale czarna:

library(ggplot2)
library(maps)
#load us map data
all_states <- map_data("state")
#plot all states with ggplot
p <- ggplot()
p <- p + geom_polygon( data=all_states, aes(x=long, y=lat, group = group),colour="white", fill="grey10" )
p

prosloan$region <- prosloan$state

# to nie dziala bo nie ma long
#p <- ggplot()
#p <- p + geom_polygon( data=all+states, aes(x=long, y=lat, group = group),colour="white" )
#p <- p + geom_jitter( data=prosloan, position=position_jitter(width=0.5, height=0.5), aes(x=long, y=lat, size = enrollment,color=state)) + scale_size(name="IncomeRange")
#p <- p + geom_text( data=prosloan, hjust=0.5, vjust=-0.5, aes(x=longitude, y=lat, label=label), colour="gold2", size=4 )
#p

display.brewer.all()
brewer.pal(4, 'Set3')
#dlaczego to nie dziala?
ggplot(all_states,aes(x=long, y=lat, group=group, fill=IncomeRange) +
  scale_fill_gradientn("",colours=brewer.pal(9,"YlOrRd"))+
  geom_polygon()+coord_map()+
  labs(title="Income by State",x="",y="")+
  theme_bw()

names(data.frame())
# to jest po prostu czarne
ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) + 
  geom_bar(color="black") + ggtitle('Is Borrower Home Owner') + 
  xlab('Is Borrower Home Owner') + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) + 
  ylab('Count') + 
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))



ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) + 
  geom_bar(color="black", aes(fill = "#"999999")) + 
  ggtitle('Is Borrower Home Owner') + 
  xlab('Is Borrower Home Owner') + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) + 
  ylab('Count') + 
  scale_fill_manual(values=c("#E69F00", #999999", "#E69F00", "#56B4E9","#999999", "#E69F00", "#56B4E9","#999999", "#E69F00"))




prosloan$region <- tolower(state.name[match(prosloan$BorrowerState,state.abb)])
prosloan$region
total <- merge(all_states, prosloan, by="region")
total <- merge(prosloan, all_states, by ="region")

p <- ggplot()
p <- p + geom_polygon(total(x=long, y=lat, group = group),colour="white" )
p <- p + geom_point( df, aes(x=long, y=lat), color="coral1") + scale_size(name="Total enrollment")
p <- p + geom_text( df, hjust=0.5, vjust=-0.5, aes(x=long, y=lat, label=label), colour="gold2", size=4 )
p

total <- merge(all_states, prosloan, by="region")
head(all_states)
head(prosloan)



# Borrower Profile - Employment Status ~ LoanOriginalAmount
ggplot(aes(x = EmploymentStatus, y = LoanOriginalAmount), data = na.omit(prosloan)) +
  geom_boxplot() +
  scale_y_continuous(limits = c(0,15000)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) + 
  geom_bar(color="black", fill = ) + 
  ggtitle('Is Borrower Home Owner') + 
  xlab('Is Borrower Home Owner') + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) + 
  ylab('Count') + 
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))




ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) + 
  geom_bar(color="black", aes(fill = IsBorrowerHomeowner)) + 
  ggtitle('Is Borrower Home Owner') + 
  xlab('Is Borrower Home Owner') + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) + 
  ylab('Count')+ 
  scale_fill_brewer(palette="Dark2")

# moge zrobic wiecej heatmap bo sa fajne, np dla borrowerstate i kasa tez bedzie dobzre wygladc

## 14.  Most common term loan
prosloan$Term <- factor(prosloan$Term)
ggplot(data = prosloan, aes(x = Term, ymax = max(..count..))) + 
  geom_bar(fill='#007EE5') +
  stat_bin(geom="text", size = 4, aes(label=..count.., vjust=-0.9)) +
  ggtitle("Length of the Loan")



##26. Investor Profile - LenderYield ~ Term  to jest moje
ggplot(aes(y = EstimatedLoss, x = Term), data = prosloan) +
  geom_boxplot()

##26. Investor Profile - LenderYield ~ Term  to jest moje tez
ggplot(aes(y = EstimatedReturn, x = Term), data = prosloan) +
  geom_boxplot()

ggplot(prosloan, aes(as.factor(LoanOriginationDate.year))) +
  geom_histogram(color = 'black', fill = '#007EE5') +
  ggtitle('Number of Loans by Year') +
  stat_bin(geom="text", aes(label=..count.., vjust=-0.9, hjust=0.5)) +
  xlab('Year') +
  ylab('Number of Loans')

````


to sie powtarzalo w moim dobrym pliku
#dziwne ale tutaj nie ma nic
```{r echo=FALSE, Univariate_Plots}
ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) + 
  geom_bar(color="black", aes(fill = "#FB8072")) + 
  ggtitle('Is Borrower Home Owner') + 
  xlab('Is Borrower Home Owner') + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) + 
  ylab('Count') + 
  scale_fill_manual(values=c("#E69F00", "#FB8072",  "#E69F00", "#56B4E9","#E69F00", "#E69F00", "#56B4E9","#E69F00", "#E69F00"))
      

to co usunelam:
  
  
  ```{r echo=FALSE, Univariate_Plots}
ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) +
  geom_bar(color="black", fill = '#007EE5') +
  ggtitle('Is Borrower Home Owner') +
  xlab('Is Borrower Home Owner') +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) +
  ylab('Count')
```
  
  
   
  
  #to nie dziala, nie rozuiem czemu, nie jest to raczej potrzebne
  ```{r echo=FALSE, Univariate_Plots}
ggplot(data = prosloan, aes(x = IsBorrowerHomeowner)) + 
  geom_bar(color="black", fill = ) + 
  ggtitle('Is Borrower Home Owner') + 
  xlab('Is Borrower Home Owner') + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) + 
  ylab('Count') + 
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))
```  
  
  
  ```{r echo=FALSE, Load_the_Data}
ggplot(data = na.omit(prosloan), aes(ProsperRating)) +
  geom_bar(aes(fill = ProsperRating)) +
  ggtitle('Numbers of Loans by Prosper Rating') +
  xlab('Rating') +
  ylab('Number of Loans')
```

From all our loans a big count is already completed, 
# quite many was cancelled aswell
```{r echo=FALSE, Univariate_Plots}
ggplot(data = prosloan, aes(x = LoanStatus)) +
  geom_bar(color="black", fill = '#007EE5') +
  ggtitle('Loan Status') +
  xlab('Loan Status') +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.6)) +
  ylab('Count')
```