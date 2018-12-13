install.packages("ldatuning")
install.packages("devtools")
devtools::install_github("nikita-moor/ldatuning")
library("ldatuning")
library("topicmodels")

data("AssociatedPress", package="topicmodels")
dtm <- AssociatedPress[1:10, ]

result <- FindTopicsNumber(
  dtm,
  topics = seq(from = 5, to = 200, by = 5),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 10L,
  verbose = TRUE
)

FindTopicsNumber_plot(result)
