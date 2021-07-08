args = commandArgs(trailingOnly=TRUE)
node <- as.numeric(args[1])
global_fits <- args[2]
batch_size <- as.numeric(args[3])
n_batches <- as.numeric(args[4])
seed <- as.numeric(args[5])
sim_length <- as.numeric(args[6])
out_dir <- args[7]

print(paste0('beginning node ', node))

fits <- readRDS(global_fits)

set.seed(seed)
n <- n_batches * batch_size
seasonality <- do.call(rbind, fits$seasonality)
params <- data.frame(
  average_age = runif(n, 20 * 365, 40 * 365),
  init_EIR = runif(n, 0, 100),
  Q0 = runif(n, 0, 1)
)
params <- cbind(
  params,
  seasonality[sample(nrow(seasonality), n, replace = TRUE), ]
)

process_row <- function(row) {
  ICDMM::run_model(
    ssa0 = row$seasonal_a0,
    ssa1 = row$seasonal_a1,
    ssa2 = row$seasonal_a2,
    ssa3 = row$seasonal_a3,
    ssb1 = row$seasonal_b1,
    ssb2 = row$seasonal_b2,
    ssb3 = row$seasonal_b3,
    eta  = 1 / row$average_age,
    Q0   = row$Q0,
    time = sim_length,
    init_EIR = row$init_EIR
  )$prev
}

batches <- split(
  seq(nrow(params)),
  (seq(nrow(params))-1) %/% batch_size
)

for (batch_i in seq_along(batches)) {
  outpath <- file.path(
    out_dir,
    paste0('realisation_', node, '_batch_', batch_i, '.json')
  )
  if (!file.exists(outpath)) {
    start_time <- Sys.time()
    print(paste0('node ', node, ' batch ', batch_i, ' starting'))
    # do the work
    results <- lapply(
      batches[[batch_i]],
      function (i) process_row(params[i,])
    )

    outdata <- params[batches[[batch_i]],]
    outdata$prev <- results

    jsonlite::write_json(outdata, outpath, pretty=TRUE)
    print(paste0('node ', node, ' batch ', batch_i, ' completed'))
    print(Sys.time())
    print(Sys.time() - start_time)
  }
}
