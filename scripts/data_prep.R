library(foresite)

# Clean intervention data
interventions <- data.frame(data.table::rbindlist(
  lapply(sites, function(s) get(s)$interventions),
  fill=T,
  use.names=T
))
demography <- data.table::rbindlist(
  lapply(sites, function(s) get(s)$demography),
  fill=T,
  use.names=T
)
vectors <- data.table::rbindlist(
  lapply(sites, function(s) get(s)$vectors),
  fill=T,
  use.names=T
)
seasonality <- data.table::rbindlist(
  lapply(sites, function(s) get(s)$seasonality),
  fill=T,
  use.names=T
)

#NOTE: there is a bug in foresite interventions where Yemeni sites have duplicate entries
interventions <- interventions[interventions$iso3c != 'YEM',]

vectors <- data.frame(subset(vectors, vectors$species %in% c("gambiae", "arabiensis", "funestus")))
included_sites <- vectors[c('iso3c', 'name_1')]
included_sites <- included_sites[!duplicated(included_sites),]
demography <- subset(demography, demography$iso3c %in% unique(vectors$iso3c))
interventions <- merge(data.frame(interventions), included_sites)
seasonality <- merge(data.frame(seasonality[is.na(seasonality$name_2),]), included_sites)

write.csv(vectors, 'vectors.csv', row.names = FALSE)
write.csv(demography, 'demography.csv', row.names = FALSE)
write.csv(interventions, 'interventions.csv', row.names = FALSE)
write.csv(seasonality, 'seasonality.csv', row.names = FALSE)

library(sp)
# Clean observations
observations <- read.csv('./battle_observations.csv')
shapes <- do.call(rbind, lapply(unique(vectors$iso3c), function(s) malariaAtlas::getShp(ISO = s, admin_level = "admin1")))
coordinates(observations) <- ~LONG + LAT
proj4string(observations) <- proj4string(shapes)
matches <- over(observations, shapes)
observations$iso3c <- matches$iso
observations$name_1 <- matches$name_1
observations <- merge(data.frame(observations), included_sites)
observations <- observations[is.na(observations$EXCLUSION),]
observations <- observations[observations$INTERVENTION == 'None',]
observations <- observations[observations$SPECIES == 'Pf',]
observations <- observations[!duplicated(observations),]

# Clean incidence observations
inc_names <- c('iso3c', 'name_1', 'START_YEAR', 'START_MONTH', 'END_YEAR', 'END_MONTH',
                 'FREQ_ACD_NUM', 'INC', 'INC_LAR', 'INC_UAR')
inc_observations <- observations[inc_names]
inc_observations <- aggregate(
  x = inc_observations[c('INC')],
  by= inc_observations[c('iso3c', 'name_1', 'START_YEAR', 'START_MONTH', 'END_YEAR', 'END_MONTH',
                         'FREQ_ACD_NUM', 'INC_LAR', 'INC_UAR')],
  FUN=mean
)

# Clean prevalence observations
prev_names <- c('iso3c', 'name_1', 'N', 'N_POS', 'PR_LAR', 'PR_UAR', 'PR_YEAR')
prev_observations <- observations[prev_names]
prev_observations <- prev_observations[!is.na(prev_observations$PR_YEAR),]
prev_observations <- prev_observations[!(is.na(prev_observations$N) | is.na(prev_observations$N_POS)),]
prev_years <- lapply(strsplit(prev_observations$PR_YEAR, '[-,]'), function(years) as.numeric(years))
prev_observations$START_YEAR <- as.numeric(lapply(prev_years, min))
prev_observations$END_YEAR <- as.numeric(lapply(prev_years, max))
prev_observations$PR_YEAR <- NULL
prev_observations <- aggregate(
  x = prev_observations[c('N', 'N_POS')],
  by= prev_observations[c('iso3c', 'name_1', 'PR_LAR', 'PR_UAR', 'START_YEAR', 'END_YEAR')],
  FUN=sum
)

write.csv(inc_observations, 'inc.csv', row.names = FALSE)
write.csv(prev_observations, 'prev.csv', row.names = FALSE)
