sigma <- function(x){
  return(1/(1+exp(-x)))
}

simulation_cla <- function(n, E){
  
  Age <- rpois(n,65+0.5*E)
  
  Eth <- sample(c(0,1),n,replace=T,prob=c(0.7+0.025*E,0.3-0.025*E))
  
  MI <- sample(c(0,1),n,replace=T,prob=c(0.8,0.2))
  
  Ang <- sapply(0.2*E-0.5+1.3*MI, function(x) 
    sample(c(1,0),1,replace=T,prob=c(sigma(x),1-sigma(x))))
  
  ACE <- sapply(0.3*E-1+0.015*Age+0.001*Eth+1.5*MI, function(x) 
    sample(c(1,0),1,replace=T,prob=c(sigma(x),1-sigma(x))))
  
  NYHA1 <- sample(c(0,1),n,replace=T,prob=c(1-0.3+0.015*E,0.3-0.015*E))
  
  NYHA2 <- sapply(NYHA1, function(x) 
  {ifelse(x==1,0,sample(c(1,0),1,replace=T,prob=c(0.3,0.7)))})
  
  NYHA3 <- sapply(NYHA1+NYHA2, function(x) 
  {ifelse(x==1,0,sample(c(1,0),1,replace=T,prob=c(0.6,0.4)))})
  
  Surv <- sapply(0.4*E+1.5-0.1*(Age-65)-0.05*Eth-1.75*MI-2.5*Ang+0.6*ACE
                 +0.25*NYHA1-0.75*NYHA2-2*NYHA3, 
                 function(x) rlnorm(1, meanlog=x, sdlog=1.5))
  
  # Surv <- sapply(Surv, function(x) ifelse(x<=5, 0, 1))
  
  data <- data.frame(cbind(Age,Eth,Ang,MI,ACE,NYHA1,NYHA2,
                           NYHA3,Surv))
  return(data)
}

N = c(250, 500, 1000, 2000, 4000)

## generate data for classification task
for (i in 1:5) {
  cla <- simulation_cla(N[i],0)
  
  ## output data
  write.csv(cla,file=paste("simulate",N[i],".csv",sep=""),row.names=FALSE)
}
