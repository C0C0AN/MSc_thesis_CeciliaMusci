
require(dplyr)
require(lme4)
require(lmerTest)
require(emmeans)

# these are the musicians
musicians <- c('sub03', 'sub04', 'sub07', 'sub17', 'sub18', 'sub21' )

root_path <- '/Volumes/ceccmusci/ga_trial/epochs_for_r/'
setwd(root_path)

paths <- dir('./', pattern = '*epo.txt', full.names = T)
names(paths) = basename(paths)

epo_data <- lapply(paths, read.table, header=T,
                   sep=',')

epo_data <- dplyr::bind_rows(epo_data, .id = 'id')
epo_data$id <- gsub(epo_data$id, patter='_tnac_epo.txt', replacement = '')

epo_data <- epo_data %>% 
  select(id, condition, epoch, time, FC1) %>%
  mutate(group = ifelse(id %in% musicians, 'musician', 'non-musician'))

epoch_ave <- epo_data %>% 
  #filter(condition == 'music') %>%
  #mutate(condition = factor(condition)) %>%
  group_by(id, condition, epoch, group) %>%
  summarise(mean_amp = mean(FC1))

mod_ave <- lmer(data = epoch_ave, mean_amp ~ condition * group + (1|id))
anova(mod_ave)

emmip(mod_ave, condition ~ group, CIs = T)
emmip(mod_ave, group ~ condition, CIs = T)

emm_options(lmerTest.limit = 12000)
group_means <- emmeans(mod_ave, ~ group, lmer.df = 'satterthwaite')
pairs(group_means)

cond_means <- emmeans(mod_ave, ~ condition, lmer.df = 'satterthwaite')
pairs(cond_means, adjust='fdr')