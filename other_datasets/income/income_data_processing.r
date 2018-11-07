adult <- read.csv("adult_income_test_nospace.csv", sep = ",")

str(adult)

# combining
adult$workclass <- as.character(adult$workclass)

adult$workclass[adult$workclass == "Without-pay" | 
                  adult$workclass == "Never-worked"] <- "Unemployed"

adult$workclass[adult$workclass == "State-gov" |
                  adult$workclass == "Local-gov"] <- "SL-gov"

adult$workclass[adult$workclass == "Self-emp-inc" |
                  adult$workclass == "Self-emp-not-inc"] <- "Self-employed"

adult$marital.status <- as.character(adult$marital.status)

adult$marital.status[adult$marital.status == "Married-AF-spouse" |
                       adult$marital.status == "Married-civ-spouse" |
                       adult$marital.status == "Married-spouse-absent"] <- "Married"

adult$marital.status[adult$marital.status == "Divorced" |
                       adult$marital.status == "Separated" |
                       adult$marital.status == "Widowed"] <- "Not-Married"

adult$native.country <- as.character(adult$native.country)

north.america <- c("Canada", "Cuba", "Dominican-Republic", "El-Salvador", "Guatemala",
                   "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua",
                   "Outlying-US(Guam-USVI-etc)", "Puerto-Rico", "Trinadad&Tobago",
                   "United-States")
asia <- c("Cambodia", "China", "Hong", "India", "Iran", "Japan", "Laos",
          "Philippines", "Taiwan", "Thailand", "Vietnam")
south.america <- c("Columbia", "Ecuador", "Peru")
europe <- c("England", "France", "Germany", "Greece", "Holand-Netherlands",
            "Hungary", "Ireland", "Italy", "Poland", "Portugal", "Scotland",
            "Yugoslavia")
other <- c("South", "?")

adult$native.country[adult$native.country %in% north.america] <- "North America"
adult$native.country[adult$native.country %in% asia] <- "Asia"
adult$native.country[adult$native.country %in% south.america] <- "South America"
adult$native.country[adult$native.country %in% europe] <- "Europe"
adult$native.country[adult$native.country %in% other] <- "Other"

adult$native.country <- as.factor(adult$native.country)
adult$marital.status <- as.factor(adult$marital.status)
adult$workclass <- as.factor(adult$workclass)

adult[adult == "?"] <- NA

adult <- na.omit(adult)

str(adult)

adultmatrix <- data.matrix(adult)

write.csv(adultmatrix,file="adult_income_test_clean.csv",row.names = FALSE,col.names = FALSE)
