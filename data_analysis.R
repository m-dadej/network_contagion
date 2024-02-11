library(tidyverse)
library(ggplot2)
library(ggridges)

df <- read.csv("C:/Users/HP/Documents/julia/finansowe/contagion/data/results_done.csv")

df <- rename(df,"sigma" = "σ", "sigma_ss" = "σ_ss")

prob <- function(x, from, to){
  
  def_probs <- rep(0, to - from)
  
    for (i in from:to) {
      def_probs[i-1] <- sum(x > i) / sum(x > 0)
    }
  return(def_probs)
}
  

select(df, sigma, sigma_ss, n_default) %>%
  filter(sigma == 4.001 & sigma_ss <= -2) %>%
  #mutate(sigma_ss = round(sigma_ss, digits = 0)) %>%
  group_by(sigma_ss) %>%
  summarise(prob = list(prob(n_default, 2, 50))) %>%
  unnest_wider(prob, names_sep = "_") %>%
  pivot_longer(cols = -sigma_ss) %>%
  mutate(n = as.numeric(str_remove_all(name, "prob_"))) %>%
  ggplot(aes(x = sigma_ss, y = value)) +
  geom_line(aes(color = n, group = name)) 


# P(n_default > k | sigma_ss = i) - P(n_default > k | sigma_ss = 0)
select(df, sigma, sigma_ss, n_default) %>%
  filter(sigma == 4.001) %>%
  #mutate(sigma_ss = round(sigma_ss, digits = 0)) %>%
  group_by(sigma_ss) %>%
  summarise(prob = list(prob(n_default, 2, 50))) %>%
  unnest_wider(prob, names_sep = "_") %>%
  pivot_longer(cols = -sigma_ss) %>%
  mutate(n = as.numeric(str_remove_all(name, "prob_"))) %>%
  group_by(n) %>%
  summarise(sigma_ss, diff = value - value[sigma_ss == 0]) %>%
  ungroup() %>%
  ggplot(aes(x = sigma_ss, y = diff)) +
  geom_line() +
  facet_wrap(~n, scales = "free_y")
  
select(df, sigma, sigma_ss, n_default) %>%
  filter(sigma == 4.001 & sigma_ss <= -2) %>%
  mutate(sigma_ss = round(sigma_ss, digits = 0)) %>%
  ggplot(aes(x = n_default, y = sigma_ss, group  = sigma_ss)) + 
  geom_density_ridges() +
  scale_x_log10()


filter(df, sigma == 4.001) %>%
  #mutate(sigma_ss = round(sigma_ss, digits = 0)) %>%
  group_by(sigma_ss) %>%
  summarise(m_interm = mean(interm)) %>%
  ggplot(aes(x = sigma_ss, y = m_interm)) +
  geom_line()

filter(df, n_default >= 10) %>%
  summarise(mean(n_default))

select(df, sigma, sigma_ss, n_default) %>%
  filter(sigma == 4.001) %>%
  #mutate(sigma_ss = round(sigma_ss, digits = 0)) %>%
  group_by(sigma_ss) %>%
  summarise(prob = sum(n_default > 20) / sum(n_default > 0)) %>%
  #summarise(prob = mean(n_default)) %>%
  ungroup() %>%
  ggplot(aes(x = sigma_ss, y = prob)) +
  geom_line()




sample_frac(df, 0.1) %>%
ggplot(aes(x = n_default, y = sigma_ss)) +
  geom_point()
