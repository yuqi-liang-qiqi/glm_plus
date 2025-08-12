#' Bayesian quantile regression in the OR2 model
#'
#' This function estimates Bayesian quantile regression in the OR2 model (ordinal quantile model with
#' exactly 3 outcomes) and reports the posterior mean, posterior standard deviation, 95
#' percent posterior credible intervals and inefficiency factor of \eqn{(\beta, \sigma)}. The output also displays the log of
#' marginal likelihood and the DIC.
#'
#' @usage quantregOR2(y, x, b0, B0 , n0, d0, gammacp2, burn, mcmc, p, accutoff, maxlags, verbose)
#'
#' @param y         observed ordinal outcomes, column vector of size \eqn{(n x 1)}.
#' @param x         covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#' @param b0        prior mean for \eqn{\beta}.
#' @param B0        prior covariance matrix for \eqn{\beta}.
#' @param n0        prior shape parameter of the inverse-gamma distribution for \eqn{\sigma}, default is 5.
#' @param d0        prior scale parameter of the inverse-gamma distribution for \eqn{\sigma}, default is 8.
#' @param gammacp2  one and only cut-point other than 0, default is 3.
#' @param burn      number of burn-in MCMC iterations.
#' @param mcmc      number of MCMC iterations, post burn-in.
#' @param p         quantile level or skewness parameter, p in (0,1).
#' @param accutoff  autocorrelation cut-off to identify the number of lags and form batches to compute the inefficiency factor, default is 0.05.
#' @param maxlags   maximum lag at which to calculate the acf in inefficiency factor calculation, default is 400.
#' @param verbose   whether to print the final output and provide additional information or not, default is TRUE.
#'
#' @details
#' This function estimates Bayesian quantile regression for the
#' OR2 model using a Gibbs sampling procedure. The function takes the prior distributions
#' and other information as inputs and then iteratively samples \eqn{\beta}, \eqn{\sigma},
#' latent weight \eqn{\nu}, and latent variable z from their respective
#' conditional distributions.
#'
#' The function also provides the logarithm of marginal likelihood and the DIC. These
#' quantities can be utilized to compare two or more competing models at the same quantile.
#' The model with a higher (lower) log marginal likelihood (DIC) provides a
#' better model fit.
#'
#' @return Returns a bqrorOR2 object with components
#' \item{\code{summary}: }{summary of the MCMC draws.}
#' \item{\code{postMeanbeta}: }{posterior mean of \eqn{\beta} from the complete Gibbs run.}
#' \item{\code{postMeansigma}: }{posterior mean of \eqn{\sigma} from the complete Gibbs run.}
#' \item{\code{postStdbeta}: }{posterior standard deviation of \eqn{\beta} from the complete Gibbs run.}
#'  \item{\code{postStdsigma}: }{posterior standard deviation of \eqn{\sigma} from the complete Gibbs run.}
#'  \item{\code{dicQuant}: }{all quantities of DIC.}
#'  \item{\code{logMargLike}: }{an estimate of log marginal likelihood.}
#'  \item{\code{ineffactor}: }{inefficiency factor for each component of \eqn{\beta} and \eqn{\sigma}.}
#'  \item{\code{betadraws}: }{dataframe of the \eqn{\beta} draws from the complete Gibbs run, size is \eqn{(k x nsim)}.}
#'  \item{\code{sigmadraws}: }{dataframe of the \eqn{\sigma} draws from the complete Gibbs run, size is \eqn{(1 x nsim)}.}
#'
#' @references Rahman, M. A. (2016). `"Bayesian Quantile Regression for Ordinal Models."`
#' Bayesian Analysis, 11(1): 1-24.  DOI: 10.1214/15-BA939
#'
#' @importFrom "stats" "sd"
#' @importFrom "stats" "quantile"
#' @importFrom "pracma" "inv"
#' @importFrom "progress" "progress_bar"
#' @seealso \link[stats]{rnorm}, \link[stats]{qnorm},
#' Gibbs sampling
#' @examples
#' set.seed(101)
#' data("data25j3")
#' y <- data25j3$y
#' xMat <- data25j3$x
#' k <- dim(xMat)[2]
#' b0 <- array(rep(0, k), dim = c(k, 1))
#' B0 <- 10*diag(k)
#' n0 <- 5
#' d0 <- 8
#' output <- quantregOR2(y = y, x = xMat, b0, B0, n0, d0, gammacp2 = 3,
#' burn = 10, mcmc = 40, p = 0.25, accutoff = 0.5, maxlags = 400, verbose = TRUE)
#'
#' # Summary of MCMC draws :
#'
#' #            Post Mean Post Std Upper Credible Lower Credible Inef Factor
#' #    beta_1   -4.5185   0.9837        -3.1726        -6.2000     1.5686
#' #    beta_2    6.1825   0.9166         7.6179         4.8619     1.5240
#' #    beta_3    5.2984   0.9653         6.9954         4.1619     1.4807
#' #    sigma     1.0879   0.2073         1.5670         0.8436     2.4228
#'
#' # Log of Marginal Likelihood: -404.57
#' # DIC: 801.82
#'
#' @export
quantregOR2 <- function(y, x, b0, B0 , n0 = 5, d0 = 8, gammacp2 = 3, burn, mcmc, p, accutoff = 0.5, maxlags = 400, verbose = TRUE) {
    cols <- colnames(x)
    names(x) <- NULL
    names(y) <- NULL
    x <- as.matrix(x)
    y <- as.matrix(y)
    if ( dim(y)[2] != 1){
        stop("input y should be a column vector")
    }
    if ( any(!all(y == floor(y)))){
        stop("each entry of y must be an integer")
    }
    if ( !all(is.numeric(x))){
        stop("each entry in x must be numeric")
    }
    if ( length(mcmc) != 1){
        stop("parameter mcmc must be scalar")
    }
    if ( !is.numeric(mcmc)){
        stop("parameter mcmc must be a numeric")
    }
    if ( !is.numeric(burn)){
        stop("parameter burn must be a numeric")
    }
    if ( length(gammacp2) != 1){
        stop("parameter gammacp2 must be scalar")
    }
    if ( !is.numeric(gammacp2)){
        stop("parameter gammacp2 must be a numeric")
    }
    if ( length(p) != 1){
        stop("parameter p must be scalar")
    }
    if ( any(p < 0 | p > 1)){
        stop("parameter p must be between 0 to 1")
    }
    if ( length(n0) != 1){
        stop("parameter n0 must be scalar")
    }
    if ( !all(is.numeric(n0))){
        stop("parameter n0 must be numeric")
    }
    if ( length(d0) != 1){
        stop("parameter d0 must be scalar")
    }
    if ( !all(is.numeric(d0))){
        stop("parameter d0 must be numeric")
    }
    J <- dim(as.array(unique(y)))[1]
    if ( J > 3 ){
        stop("This function is for exactly 3 outcome
                variables. Please correctly specify the inputs
             to use quantregOR2")
    }
    n <- dim(x)[1]
    k <- dim(x)[2]
    if ((dim(B0)[1] != (k)) | (dim(B0)[2] != (k))){
        stop("B0 is the prior variance to sample beta
             must have size kxk")
    }
    nsim <- burn + mcmc

    invB0 <- inv(B0)
    invB0b0 <- invB0 %*% b0

    beta <- array(0, dim = c(k, nsim))
    sigma <- array(0, dim = c(1, nsim))
    btildeStore <- array(0, dim = c(k, nsim))
    BtildeStore <- array(0, dim = c(k, k, nsim))

    beta[, 1] <- array(rep(0,k), dim = c(k, 1))
    sigma[1] <- 2
    nu <- array(5 * rep(1,n), dim = c(n, 1))
    gammacp <- array(c(-Inf, 0, gammacp2, Inf), dim = c(1, J+1))
    indexp <- 0.5
    theta <- (1 - 2 * p) / (p * (1 - p))
    tau <- sqrt(2 / (p * (1 - p)))
    tau2 <- tau^2

    z <- array( (rnorm(n, mean = 0, sd = 1)), dim = c(n, 1))
    if(verbose) {
        pb <- progress_bar$new(" Simulation in Progress [:bar] :percent",
                           total = nsim, clear = FALSE, width = 100)
    }

    for (i in 2:nsim) {
        betadraw <- drawbetaOR2(z, x, sigma[(i - 1)], nu, tau2, theta, invB0, invB0b0)
        beta[, i] <- betadraw$beta
        btildeStore[, i] <- betadraw$btilde
        BtildeStore[, , i] <- betadraw$Btilde

        sigmadraw <- drawsigmaOR2(z, x, beta[, i], nu, tau2, theta, n0, d0)
        sigma[i] <- sigmadraw$sigma

        nu <- drawnuOR2(z, x, beta[, i], sigma[i], tau2, theta, indexp)

        z <- drawlatentOR2(y, x, beta[, i], sigma[i], nu, theta, tau2, gammacp)
        if(verbose) {
            pb$tick()
        }
    }

    postMeanbeta <- rowMeans(beta[, (burn + 1):nsim])
    postStdbeta <- apply(beta[, (burn + 1):nsim], 1, sd)
    postMeansigma <- mean(sigma[(burn + 1):nsim])
    postStdsigma <- std(sigma[(burn + 1):nsim])

    dicQuant <- dicOR2(y, x, beta, sigma, gammacp,
                            postMeanbeta, postMeansigma, burn, mcmc, p)

    logMargLike <- logMargLikeOR2(y, x, b0, B0,
                                                n0, d0, postMeanbeta,
                                                postMeansigma, btildeStore,
                                                BtildeStore, gammacp2, p, verbose)

    ineffactor <- ineffactorOR2(x, beta, sigma, accutoff, maxlags, FALSE)

    postMeanbeta <- array(postMeanbeta, dim = c(k, 1))
    postStdbeta <- array(postStdbeta, dim = c(k, 1))
    postMeansigma <- array(postMeansigma)
    postStdsigma <- array(postStdsigma)

    upperCrediblebeta <- array(apply(beta[ ,(burn + 1):nsim], 1, quantile, c(0.975)), dim = c(k, 1))
    lowerCrediblebeta <- array(apply(beta[ ,(burn + 1):nsim], 1, quantile, c(0.025)), dim = c(k, 1))
    upperCrediblesigma <- quantile(sigma[(burn + 1):nsim], c(0.975))
    lowerCrediblesigma <- quantile(sigma[(burn + 1):nsim], c(0.025))

    inefficiencyBeta <- array(ineffactor[1:k], dim = c(k,1))
    inefficiencySigma <- array(ineffactor[k+1])

    allQuantbeta <- cbind(postMeanbeta, postStdbeta, upperCrediblebeta, lowerCrediblebeta, inefficiencyBeta)
    allQuantsigma <- cbind(postMeansigma, postStdsigma, upperCrediblesigma, lowerCrediblesigma, inefficiencySigma)
    summary <- rbind(allQuantbeta, allQuantsigma)
    name <- list('Post Mean', 'Post Std', 'Upper Credible', 'Lower Credible','Inef Factor')
    dimnames(summary)[[2]] <- name
    dimnames(summary)[[1]] <- letters[1:(k+1)]
    dimnames(beta)[[1]] <- letters[1:k]
    dimnames(sigma)[[1]] <- letters[1]
    j <- 1
    if (is.null(cols)) {
        rownames(summary)[j] <- c('Intercept')
        for (i in paste0("beta_",1:k)) {
            rownames(summary)[j] = i
            rownames(beta)[j] = i
            j = j + 1
        }
    }
    else {
        for (i in cols) {
            rownames(summary)[j] = i
            rownames(beta)[j] = i
            j = j + 1
        }
    }
    rownames(summary)[j] <- 'sigma'
    rownames(sigma)[1] <- 'sigma'

    if (verbose) {
        print(noquote('Summary of MCMC draws : '))
        cat("\n")
        print(round(summary, 4))
        cat("\n")
        print(noquote(paste0('Log of Marginal Likelihood: ', round(logMargLike, 2))))
        print(noquote(paste0('DIC: ', round(dicQuant$DIC, 2))))
    }

    beta <- data.frame(beta)
    sigma <- data.frame(sigma)

    result <- list("summary" = summary,
                   "postMeanbeta" = postMeanbeta,
                   "postStdbeta" = postStdbeta,
                   "postMeansigma" = postMeansigma,
                   "postStdsigma" = postStdsigma,
                   "dicQuant" = dicQuant,
                   "logMargLike" = logMargLike,
                   "ineffactor" = ineffactor,
                   "betadraws" = beta,
                   "sigmadraws" = sigma)

    class(result) <- "bqror2"
    return(result)
}
#' Samples latent variable z in the OR2 model
#'
#' This function samples the latent variable z from a univariate truncated
#' normal distribution in the OR2 model (ordinal quantile model with exactly 3 outcomes).
#'
#' @usage drawlatentOR2(y, x, beta, sigma, nu, theta, tau2, gammacp)
#'
#' @param y         observed ordinal outcomes, column vector of size \eqn{(n x 1)}.
#' @param x         covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#' @param beta      Gibbs draw of \eqn{\beta}, a column vector of size \eqn{(k x 1)}.
#' @param sigma     \eqn{\sigma}, a scalar value.
#' @param nu        modified latent weight, column vector of size \eqn{(n x 1)}.
#' @param tau2      2/(p(1-p)).
#' @param theta     (1-2p)/(p(1-p)).
#' @param gammacp   row vector of cut-points including -Inf and Inf.
#'
#' @details
#' This function samples the latent variable z from a univariate truncated normal
#' distribution.
#'
#' @return latent variable z of size \eqn{(n x 1)} from a univariate truncated distribution.
#'
#' @references Albert, J., and Chib, S. (1993). `"Bayesian Analysis of Binary and Polychotomous Response Data."`
#' Journal of the American Statistical Association, 88(422): 669`-`679. DOI: 10.1080/01621459.1993.10476321
#'
#' Devroye, L. (2014). `"Random variate generation for the generalized inverse Gaussian distribution."`
#' Statistics and Computing, 24(2): 239`-`246. DOI: 10.1007/s11222-012-9367-z
#'
#' @seealso Gibbs sampling, truncated normal distribution,
#' \link[truncnorm]{rtruncnorm}
#' @importFrom "truncnorm" "rtruncnorm"
#' @examples
#' set.seed(101)
#' data("data25j3")
#' y <- data25j3$y
#' xMat <- data25j3$x
#' beta <- c(1.810504, 1.850332, 6.181163)
#' sigma <- 0.9684741
#' n <- dim(xMat)[1]
#' nu <- array(5 * rep(1,n), dim = c(n, 1))
#' theta <- 2.6667
#' tau2 <- 10.6667
#' gammacp <- c(-Inf, 0, 3, Inf)
#' output <- drawlatentOR2(y, xMat, beta, sigma, nu,
#' theta, tau2, gammacp)
#'
#' # output
#' #   1.257096 10.46297 4.138694
#' #   28.06432 4.179275 19.21582
#' #   11.17549 13.79059 28.3650 .. soon
#'
#' @export
drawlatentOR2 <- function(y, x, beta, sigma, nu, theta, tau2, gammacp) {
    if ( dim(y)[2] != 1){
        stop("input y should be a column vector")
    }
    if ( any(!all(y == floor(y)))){
        stop("each entry of y must be an integer")
    }
    if ( !all(is.numeric(x))){
        stop("each entry in x must be numeric")
    }
    if ( !all(is.numeric(beta))){
        stop("each entry in beta must be numeric")
    }
    if ( length(sigma) != 1){
        stop("parameter sigma must be scalar")
    }
    if ( !all(is.numeric(nu))){
        stop("each entry in nu must be numeric")
    }
    if ( length(tau2) != 1){
        stop("parameter tau2 must be scalar")
    }
    if ( !all(is.numeric(tau2))){
        stop("parameter tau2 must be numeric")
    }
    if ( length(theta) != 1){
        stop("parameter theta must be scalar")
    }
    if ( !all(is.numeric(theta))){
        stop("parameter theta must be numeric")
    }
    n <- dim(y)[1]
    z <- array(0, dim = c(n, 1))
    for (i in 1:n) {
        meancomp <- (x[i, ] %*% beta) + (theta * nu[i, 1])
        std <- sqrt(tau2 * sigma * nu[i, 1])
        temp <- y[i]
        a1 <- gammacp[temp]
        b1 <- gammacp[temp + 1]
        z[i, 1] <- rtruncnorm(n = 1, a = a1, b = b1, mean = meancomp, sd = std)
    }
    return(z)
}
#' Samples \eqn{\beta} in the OR2 model
#'
#' This function samples \eqn{\beta} from its conditional
#' posterior distribution in the OR2 model (ordinal quantile model with exactly 3
#' outcomes).
#'
#' @usage drawbetaOR2(z, x, sigma, nu, tau2, theta, invB0, invB0b0)
#'
#' @param z         continuous latent values, vector of size \eqn{(n x 1)}.
#' @param x         covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#' @param sigma     \eqn{\sigma}, a scalar value.
#' @param nu        modified latent weight, column vector of size \eqn{(n x 1)}.
#' @param tau2      2/(p(1-p)).
#' @param theta     (1-2p)/(p(1-p)).
#' @param invB0     inverse of prior covariance matrix of normal distribution.
#' @param invB0b0   prior mean pre-multiplied by invB0.
#'
#' @details
#'
#' This function samples \eqn{\beta}, a vector, from its conditional posterior distribution
#' which is an updated multivariate normal distribution.
#'
#' @return Returns a list with components
#' \item{\code{beta}: }{\eqn{\beta}, a column vector of size \eqn{(k x 1)}, sampled from its
#' condtional posterior distribution.}
#' \item{\code{Btilde}: }{variance parameter for the posterior
#' multivariate normal distribution.}
#' \item{\code{btilde}: }{mean parameter for the
#' posterior multivariate normal distribution.}
#'
#' @references Rahman, M. A. (2016). `"Bayesian Quantile Regression for Ordinal Models."`
#' Bayesian Analysis, 11(1): 1-24. DOI: 10.1214/15-BA939
#'
#' @importFrom "MASS" "mvrnorm"
#' @importFrom "pracma" "inv"
#' @seealso Gibbs sampling, normal distribution
#' , \link[GIGrvg]{rgig}, \link[pracma]{inv}
#' @examples
#' set.seed(101)
#' z <- c(21.01744, 33.54702, 33.09195, -3.677646,
#'  21.06553, 1.490476, 0.9618205, -6.743081, 21.02186, 0.6950479)
#' x <- matrix(c(
#'      1, -0.3010490, 0.8012506,
#'      1,  1.2764036, 0.4658184,
#'      1,  0.6595495, 1.7563655,
#'      1, -1.5024607, -0.8251381,
#'      1, -0.9733585, 0.2980610,
#'      1, -0.2869895, -1.0130274,
#'      1,  0.3101613, -1.6260663,
#'      1, -0.7736152, -1.4987616,
#'      1,  0.9961420, 1.2965952,
#'      1, -1.1372480, 1.7537353),
#'      nrow = 10, ncol = 3, byrow = TRUE)
#' sigma <- 1.809417
#' n <- dim(x)[1]
#' nu <- array(5 * rep(1,n), dim = c(n, 1))
#' tau2 <- 10.6667
#' theta <- 2.6667
#' invB0 <- matrix(c(
#'      1, 0, 0,
#'      0, 1, 0,
#'      0, 0, 1),
#'      nrow = 3, ncol = 3, byrow = TRUE)
#' invB0b0 <- c(0, 0, 0)
#'
#' output <- drawbetaOR2(z, x, sigma, nu, tau2, theta, invB0, invB0b0)
#'
#' # output$beta
#' #   -0.74441 1.364846 0.7159231
#'
#' @export
drawbetaOR2 <- function(z, x, sigma, nu, tau2, theta, invB0, invB0b0) {
    if ( !all(is.numeric(z))){
        stop("each entry in z must be numeric")
    }
    if ( !all(is.numeric(x))){
        stop("each entry in x must be numeric")
    }
    if ( length(sigma) != 1){
        stop("parameter sigma must be scalar")
    }
    if ( !all(is.numeric(nu))){
        stop("each entry in nu must be numeric")
    }
    if ( length(tau2) != 1){
        stop("parameter tau2 must be scalar")
    }
    if ( !all(is.numeric(tau2))){
        stop("parameter tau2 must be numeric")
    }
    if ( length(theta) != 1){
        stop("parameter theta must be scalar")
    }
    if ( !all(is.numeric(theta))){
        stop("parameter theta must be numeric")
    }
    if ( !all(is.numeric(invB0))){
        stop("each entry in invB0 must be numeric")
    }
    if ( !all(is.numeric(invB0b0))){
        stop("each entry in invB0b0 must be numeric")
    }
    n <- dim(x)[1]
    k <- dim(x)[2]
    meancomp <- array(0, dim = c(n, k))
    varcomp <- array(0, dim = c(k, k, n))
    q <- array(0, dim = c(1, k))
    eye <- diag(k)
    for (i in 1:n) {
        meancomp[i, ] <- (x[i, ] * (z[i] - (theta * nu[i, 1])) ) / (tau2 * sigma * nu[i,1])
        varcomp[, , i] <- ( x[i, ] %*% t(x[i, ])) / (tau2 * sigma * nu[i,1])
    }
    Btilde <- inv(invB0 + rowSums(varcomp, dims = 2))
    btilde <- Btilde %*% (invB0b0 + colSums(meancomp))
    L <- t(chol(Btilde))
    beta <- btilde + L %*%  (mvrnorm(n = 1, mu = q, Sigma = eye))

    betaReturns <- list("beta" = beta,
                   "Btilde" = Btilde,
                   "btilde" = btilde)

    return(betaReturns)
}
#' Samples \eqn{\sigma} in the OR2 model
#'
#' This function samples \eqn{\sigma} from an inverse-gamma distribution
#' in the OR2 model (ordinal quantile model with exactly 3 outcomes).
#'
#' @usage drawsigmaOR2(z, x, beta, nu, tau2, theta, n0, d0)
#'
#' @param z         Gibbs draw of continuous latent values, a column vector of size \eqn{n x 1}.
#' @param x         covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#' @param beta      Gibbs draw of \eqn{\beta}, a column vector of size \eqn{(k x 1)}.
#' @param nu        modified latent weight, column vector of size \eqn{(n x 1)}.
#' @param tau2      2/(p(1-p)).
#' @param theta     (1-2p)/(p(1-p)).
#' @param n0        prior hyper-parameter for \eqn{\sigma}.
#' @param d0        prior hyper-parameter for \eqn{\sigma}.
#'
#' @details
#' This function samples \eqn{\sigma} from an inverse-gamma distribution.
#'
#' @return Returns a list with components
#' \item{\code{sigma}: }{\eqn{\sigma}, a scalar, sampled
#' from an inverse gamma distribution.}
#' \item{\code{dtilde}: }{scale parameter of the inverse-gamma distribution.}
#'
#' @importFrom "stats" "rgamma"
#'
#' @references Rahman, M. A. (2016). `"Bayesian Quantile Regression for Ordinal Models."`
#' Bayesian Analysis, 11(1): 1-24.  DOI: 10.1214/15-BA939
#'
#' Devroye, L. (2014). `"Random variate generation for the generalized inverse Gaussian distribution."`
#' Statistics and Computing, 24(2): 239`-`246. DOI: 10.1007/s11222-012-9367-z
#'
#' @seealso \link[stats]{rgamma}, Gibbs sampling
#' @examples
#' set.seed(101)
#' z <- c(21.01744, 33.54702, 33.09195, -3.677646,
#'  21.06553, 1.490476, 0.9618205, -6.743081, 21.02186, 0.6950479)
#' x <- matrix(c(
#'      1, -0.3010490, 0.8012506,
#'      1,  1.2764036, 0.4658184,
#'      1,  0.6595495, 1.7563655,
#'      1, -1.5024607, -0.8251381,
#'      1, -0.9733585, 0.2980610,
#'      1, -0.2869895, -1.0130274,
#'      1,  0.3101613, -1.6260663,
#'      1, -0.7736152, -1.4987616,
#'      1,  0.9961420, 1.2965952,
#'      1, -1.1372480, 1.7537353),
#'      nrow = 10, ncol = 3, byrow = TRUE)
#' beta <- c(-0.74441, 1.364846, 0.7159231)
#' n <- dim(x)[1]
#' nu <- array(5 * rep(1,n), dim = c(n, 1))
#' tau2 <- 10.6667
#' theta <- 2.6667
#' n0 <- 5
#' d0 <- 8
#' output <- drawsigmaOR2(z, x, beta, nu, tau2, theta, n0, d0)
#'
#' # output$sigma
#' #   3.749524
#'
#' @export
drawsigmaOR2 <- function(z, x, beta, nu, tau2, theta, n0, d0) {
    if ( !all(is.numeric(z))){
        stop("each entry in z must be numeric")
    }
    if ( !all(is.numeric(x))){
        stop("each entry in x must be numeric")
    }
    if ( !all(is.numeric(beta))){
        stop("each entry in beta must be numeric")
    }
    if ( !all(is.numeric(nu))){
        stop("each entry in nu must be numeric")
    }
    if ( length(tau2) != 1){
        stop("parameter tau2 must be scalar")
    }
    if ( !all(is.numeric(tau2))){
        stop("parameter tau2 must be numeric")
    }
    if ( length(theta) != 1){
        stop("parameter theta must be scalar")
    }
    if ( !all(is.numeric(theta))){
        stop("parameter theta must be numeric")
    }
    if ( length(n0) != 1){
        stop("parameter n0 must be scalar")
    }
    if ( !all(is.numeric(n0))){
        stop("parameter n0 must be numeric")
    }
    if ( length(d0) != 1){
        stop("parameter d0 must be scalar")
    }
    if ( !all(is.numeric(d0))){
        stop("parameter d0 must be numeric")
    }
    n <- dim(x)[1]
    ntilde <- n0 + (3 * n)
    temp <- array(0, dim = c(n, 1))
    for (i in 1:n) {
        temp[i, 1] <- (( z[i] - x[i, ] %*% beta - theta * nu[i, 1] )^2) / (tau2 * nu[i, 1])
    }
    dtilde <- sum(temp) + d0 + (2 * sum(nu))
    sigma <- 1/rgamma(n = 1, shape = (ntilde / 2), scale = (2 / dtilde))

    sigmaReturns <- list("sigma" = sigma,
                   "dtilde" = dtilde)
    return(sigmaReturns)
}

#' Samples scale factor \eqn{\nu} in the OR2 model
#'
#' This function samples \eqn{\nu} from a generalized inverse Gaussian (GIG)
#' distribution in the OR2 model (ordinal quantile model with exactly 3 outcomes).
#'
#' @usage drawnuOR2(z, x, beta, sigma, tau2, theta, indexp)
#'
#' @param z         Gibbs draw of continuous latent values, a column vector of size \eqn{(n x 1)}.
#' @param x         covariate matrix of size \eqn{(n x k)} including a column of ones.
#' @param beta      Gibbs draw of \eqn{\beta}, a column vector of size \eqn{(k x 1)}.
#' @param sigma     \eqn{\sigma}, a scalar value.
#' @param tau2      2/(p(1-p)).
#' @param theta     (1-2p)/(p(1-p)).
#' @param indexp    index parameter of the GIG distribution which is equal to 0.5.
#'
#' @details
#' This function samples \eqn{\nu} from a GIG
#' distribution.
#'
#' @return \eqn{\nu}, a column vector of size \eqn{(n x 1)}, sampled from a GIG distribution.
#'
#' @references  Rahman, M. A. (2016), `"Bayesian Quantile Regression for Ordinal Models."`
#' Bayesian Analysis, 11(1), 1-24. DOI: 10.1214/15-BA939
#'
#' Devroye, L. (2014). `"Random variate generation for the generalized inverse Gaussian distribution."`
#' Statistics and Computing, 24(2): 239`-`246. DOI: 10.1007/s11222-012-9367-z
#'
#' @importFrom "GIGrvg" "rgig"
#' @seealso GIGrvg, Gibbs sampling, \link[GIGrvg]{rgig}
#' @examples
#' set.seed(101)
#' z <- c(21.01744, 33.54702, 33.09195, -3.677646,
#'  21.06553, 1.490476, 0.9618205, -6.743081, 21.02186, 0.6950479)
#' x <- matrix(c(
#'      1, -0.3010490, 0.8012506,
#'      1,  1.2764036, 0.4658184,
#'      1,  0.6595495, 1.7563655,
#'      1, -1.5024607, -0.8251381,
#'      1, -0.9733585, 0.2980610,
#'      1, -0.2869895, -1.0130274,
#'      1,  0.3101613, -1.6260663,
#'      1, -0.7736152, -1.4987616,
#'      1, 0.9961420, 1.2965952,
#'      1, -1.1372480, 1.7537353),
#'      nrow = 10, ncol = 3, byrow = TRUE)
#' beta <- c(-0.74441, 1.364846, 0.7159231)
#' sigma <- 3.749524
#' tau2 <- 10.6667
#' theta <- 2.6667
#' indexp <- 0.5
#' output <- drawnuOR2(z, x, beta, sigma, tau2, theta, indexp)
#'
#' # output
#' #   5.177456 4.042261 8.950365
#' #   1.578122 6.968687 1.031987
#' #   4.13306 0.4681557 5.109653
#' #   0.1725333
#'
#' @export
drawnuOR2 <- function(z, x, beta, sigma, tau2, theta, indexp) {
    if ( !all(is.numeric(z))){
        stop("each entry in z must be numeric")
    }
    if ( !all(is.numeric(x))){
        stop("each entry in x must be numeric")
    }
    if ( !all(is.numeric(beta))){
        stop("each entry in beta must be numeric")
    }
    if ( length(sigma) != 1){
        stop("parameter sigma must be scalar")
    }
    if ( length(tau2) != 1){
        stop("parameter tau2 must be scalar")
    }
    if ( !all(is.numeric(tau2))){
        stop("parameter tau2 must be numeric")
    }
    if ( length(theta) != 1){
        stop("parameter theta must be scalar")
    }
    if ( !all(is.numeric(theta))){
        stop("parameter theta must be numeric")
    }
    if ( length(indexp) != 1){
        stop("parameter indexp must be scalar")
    }
    if ( !all(is.numeric(indexp))){
        stop("parameter indexp must be numeric")
    }
    n <- dim(x)[1]
    tildeeta <- ( (theta ^ 2) / (tau2 * sigma)) + (2 / sigma)
    tildelambda <- array(0, dim = c(n, 1))
    nu <- array(0, dim = c(n, 1))
    for (i in 1:n) {
        tildelambda[i, 1] <- ( (z[i] - x[i, ] %*%  beta)^2) / (tau2 * sigma)
        nu[i, 1] <- rgig(n = 1, lambda = indexp,
                         chi = tildelambda[i, 1],
                         psi = tildeeta)
    }
    return(nu)
}

#' Deviance Information Criterion in the OR2 model
#'
#' Function for computing the DIC in the OR2 model (ordinal quantile
#' model with exactly 3 outcomes).
#'
#' @usage dicOR2(y, x, betadraws, sigmadraws, gammacp, postMeanbeta,
#' postMeansigma, burn, mcmc, p)
#'
#' @param y              observed ordinal outcomes, column vector of size \eqn{(n x 1)}.
#' @param x              covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#' @param betadraws      dataframe of the MCMC draws of \eqn{\beta}, size is \eqn{(k x nsim)}.
#' @param sigmadraws     dataframe of the MCMC draws of \eqn{\sigma}, size is \eqn{(nsim x 1)}.
#' @param gammacp        row vector of cut-points including -Inf and Inf.
#' @param postMeanbeta   posterior mean of the MCMC draws of \eqn{\beta}.
#' @param postMeansigma  posterior mean of the MCMC draws of \eqn{\sigma}.
#' @param burn           number of burn-in MCMC iterations.
#' @param mcmc           number of MCMC iterations, post burn-in.
#' @param p              quantile level or skewness parameter, p in (0,1).
#'
#' @details
#' Deviance is -2*(log likelihood) and has an important role in
#' statistical model comparison because of its relation with Kullback-Leibler
#' information criterion.
#'
#' This function provides the DIC, which can be used to compare two or more models at the
#' same quantile. The model with a lower DIC provides a better fit.
#'
#' @return Returns a list with components
#' \deqn{DIC = 2*avgdeviance - dev}
#' \deqn{pd = avgdeviance - dev}
#' \deqn{dev = -2*(logLikelihood)}.
#'
#' @references Spiegelhalter, D. J., Best, N. G., Carlin, B. P. and Linde, A. (2002).
#' `"Bayesian Measures of Model Complexity and Fit."` Journal of the
#' Royal Statistical Society B, Part 4: 583-639. DOI: 10.1111/1467-9868.00353
#'
#' Gelman, A., Carlin, J. B., Stern, H. S., and Rubin, D. B.
#' `"Bayesian Data Analysis."` 2nd Edition, Chapman and Hall. DOI: 10.1002/sim.1856
#'
#' @seealso  decision criteria
#' @examples
#' set.seed(101)
#' data("data25j3")
#' y <- data25j3$y
#' xMat <- data25j3$x
#' k <- dim(xMat)[2]
#' b0 <- array(rep(0, k), dim = c(k, 1))
#' B0 <- 10*diag(k)
#' n0 <- 5
#' d0 <- 8
#' output <- quantregOR2(y = y, x = xMat, b0, B0, n0, d0, gammacp2 = 3,
#' burn = 10, mcmc = 40, p = 0.25, accutoff = 0.5, maxlags = 400, verbose = FALSE)
#' betadraws <- output$betadraws
#' sigmadraws <- output$sigmadraws
#' gammacp <- c(-Inf, 0, 3, Inf)
#' postMeanbeta <- output$postMeanbeta
#' postMeansigma <- output$postMeansigma
#' mcmc = 40
#' burn <- 10
#' nsim <- burn + mcmc
#' dic <- dicOR2(y, xMat, betadraws, sigmadraws, gammacp,
#' postMeanbeta, postMeansigma, burn, mcmc, p = 0.25)
#'
#' # DIC
#' #   801.8191
#' # pd
#' #   6.608594
#' # dev
#' #   788.6019
#'
#' @export
dicOR2 <- function(y, x, betadraws, sigmadraws, gammacp, postMeanbeta,
                      postMeansigma, burn, mcmc, p) {
    cols <- colnames(x)
    names(x) <- NULL
    names(y) <- NULL
    x <- as.matrix(x)
    y <- as.matrix(y)
    betadraws <- as.matrix(betadraws)
    sigmadraws <- as.matrix(sigmadraws)
    if (dim(y)[2] != 1){
        stop("input y should be a column vector")
    }
    if ( any(!all(y == floor(y)))){
        stop("each entry of y must be an integer")
    }
    if ( !all(is.numeric(x))){
        stop("each entry in x must be numeric")
    }
    if ( length(p) != 1){
        stop("parameter p must be scalar")
    }
    if (any(p < 0 | p > 1)){
        stop("parameter p must be between 0 to 1")
    }
    if ( !all(is.numeric(postMeanbeta))){
        stop("each entry in postMeanbeta must be numeric")
    }
    if ( !all(is.numeric(postMeansigma))){
        stop("each entry in postMeansigma must be numeric")
    }
    if ( !all(is.numeric(betadraws))){
        stop("each entry in betadraws must be numeric")
    }
    if ( !all(is.numeric(sigmadraws))){
        stop("each entry in sigmadraws must be numeric")
    }
    if ( length(mcmc) != 1){
        stop("parameter mcmc must be scalar")
    }
    if ( length(burn) != 1){
        stop("parameter nsim must be scalar")
    }
    nsim <- mcmc + burn
    k <- dim(x)[2]
    dev <- array(0, dim = c(1))
    DIC <- array(0, dim = c(1))
    pd <- array(0, dim = c(1))
    dev <- 2 * qrnegLogLikeOR2(y, x, gammacp, postMeanbeta, postMeansigma, p)

    postBurnin <- dim(betadraws[, (burn + 1):nsim])[2]
    Deviance <- array(0, dim = c(1, postBurnin))
    for (i in 1:postBurnin) {
        Deviance[1, i] <- 2 * qrnegLogLikeOR2(y, x, gammacp,
                                                   betadraws[ ,(burn + i)],
                                                   sigmadraws[ ,(burn + i)],
                                                   p)
    }
    avgDeviance <- mean(Deviance)
    DIC <- (2 * avgDeviance) - dev
    pd <- avgDeviance - dev
    result <- list("DIC" = DIC,
                   "pd" = pd,
                   "dev" = dev)
    return(result)
}
#' Negative sum of log-likelihood in the OR2 model
#'
#' This function computes the negative sum of log-likelihood in the OR2 model (ordinal quantile
#' model with exactly 3 outcomes).
#'
#' @param y         observed ordinal outcomes, column vector of size \eqn{(n x 1)}.
#' @param x         covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#' @param gammacp   a row vector of cutpoints including (-Inf, Inf).
#' @param betaOne   a sample draw of \eqn{\beta} of size \eqn{(k x 1)}.
#' @param sigmaOne  a sample draw of \eqn{\sigma}, a scalar value.
#' @param p         quantile level or skewness parameter, p in (0,1).
#'
#' @details
#' This function computes the negative sum of log-likelihood in the OR2 model where the error is assumed to follow
#' an AL distribution.
#'
#' @return Returns the negative sum of log-likelihood.
#'
#' @references Rahman, M. A. (2016). `"Bayesian Quantile Regression for Ordinal Models."`
#' Bayesian Analysis, 11(1): 1-24. DOI: 10.1214/15-BA939
#'
#' @seealso likelihood maximization
#' @examples
#' set.seed(101)
#' data("data25j3")
#' y <- data25j3$y
#' xMat <- data25j3$x
#' p <- 0.25
#' gammacp <- c(-Inf, 0, 3, Inf)
#' betaOne <- c(1.810504, 1.850332, 6.18116)
#' sigmaOne <- 0.9684741
#' output <- qrnegLogLikeOR2(y, xMat, gammacp, betaOne, sigmaOne, p)
#'
#' # output
#' #   902.4045
#'
#' @export
qrnegLogLikeOR2 <- function(y, x, gammacp, betaOne, sigmaOne, p) {
    cols <- colnames(x)
    names(x) <- NULL
    names(y) <- NULL
    x <- as.matrix(x)
    y <- as.matrix(y)
    if (dim(y)[2] != 1){
        stop("input y should be a column vector")
    }
    if ( any(!all(y == floor(y)))){
        stop("each entry of y must be an integer")
    }
    if ( !all(is.numeric(x))){
        stop("each entry in x must be numeric")
    }
    if ( !all(is.numeric(betaOne))){
        stop("each entry in betaOne must be numeric")
    }
    if ( length(sigmaOne) != 1){
        stop("parameter sigmaOne must be scalar")
    }
    if (any(p < 0 | p > 1)){
        stop("parameter p must be between 0 to 1")
    }
    J <- dim(unique(y))[1]
    n <- dim(y)[1]
    lnpdf <- array(0, dim = c(n, 1))
    mu <- x %*% betaOne
    for (i in 1:n) {
        meanf <- mu[i]
        if (y[i] == 1) {
            lnpdf[i] <- log(alcdf(0, meanf, sigmaOne, p))
        }
        else if (y[i] == J) {
            lnpdf[i] <- log(1 - alcdf(gammacp[J], meanf, sigmaOne, p))
        }
        else {
            w <- (alcdf(gammacp[J], meanf, sigmaOne, p) -
                      alcdf(gammacp[(J - 1)], meanf, sigmaOne, p))
            lnpdf[i] <- log(w)
        }
    }
    negsuminpdf <- -sum(lnpdf)
    return(negsuminpdf)
}

#' Generates random numbers from an AL distribution
#'
#' @description
#' This function generates a vector of random numbers from an AL
#' distribution at quantile p.
#'
#' @usage rndald(sigma, p, n)
#'
#' @param sigma  scale factor, a scalar value.
#' @param p      quantile or skewness parameter, p in (0,1).
#' @param n      number of observations
#'
#' @details
#' Generates a vector of random numbers from an AL distribution
#' as a mixture of normal`–`exponential distributions.
#'
#' @return Returns a vector \eqn{(n x 1)} of random numbers from an AL(0, \eqn{\sigma}, p)
#'
#' @references
#' Kozumi, H., and Kobayashi, G. (2011). `"Gibbs Sampling Methods for Bayesian Quantile Regression."`
#' Journal of Statistical Computation and Simulation, 81(11): 1565`-`1578. DOI: 10.1080/00949655.2010.496117
#'
#' Yu, K., and Zhang, J. (2005). `"A Three-Parameter Asymmetric Laplace Distribution."`
#' Communications in Statistics - Theory and Methods, 34(9-10), 1867`-`1879. DOI: 10.1080/03610920500199018
#'
#' @importFrom "stats" "rnorm" "rexp"
#' @seealso asymmetric Laplace distribution
#' @examples
#' set.seed(101)
#' sigma <- 2.503306
#' p <- 0.25
#' n <- 1
#' output <- rndald(sigma, p, n)
#'
#' # output
#' #   1.07328
#'
#' @export
rndald <- function(sigma, p, n){
    if ( any(p < 0 | p > 1)){
        stop("parameter p must be between 0 to 1")
    }
    if ( n != floor(n)){
        stop("parameter n must be an integer")
    }
    if ( length(sigma) != 1){
        stop("parameter sigma must be scalar")
    }
    u <- rnorm(n = n, mean = 0, sd = 1)
    w <- rexp(n = n, rate = 1)
    theta <- (1 - 2 * p) / (p * (1 - p))
    tau <- sqrt(2 / (p * (1 - p)))
    eps <- sigma * (theta * w + tau * sqrt(w) * u)
    return(eps)
}

#' Inefficiency factor in the OR2 model
#'
#' This function calculates the inefficiency factor from the MCMC draws
#' of \eqn{(\beta, \sigma)} in the OR2 model (ordinal quantile model with exactly 3 outcomes). The
#' inefficiency factor is calculated using the batch-means method.
#'
#' @usage ineffactorOR2(x, betadraws, sigmadraws, accutoff, maxlags, verbose)
#'
#' @param x                         covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#'                                  This input is used to extract column names, if available, but not used in calculation.
#' @param betadraws                 dataframe of the Gibbs draws of \eqn{\beta}, size \eqn{(k x nsim)}.
#' @param sigmadraws                dataframe of the Gibbs draws of \eqn{\sigma}, size \eqn{(1 x nsim)}.
#' @param accutoff                  cut-off to identify the number of lags and form batches, default is 0.05.
#' @param maxlags                   maximum lag at which to calculate the acf, default is 400.
#' @param verbose                   whether to print the final output and provide additional information or not, default is TRUE.
#'
#' @details
#' Calculates the inefficiency factor of \eqn{(\beta, \sigma)} using the batch-means
#' method based on the Gibbs draws. Inefficiency factor can be interpreted as the cost of
#' working with correlated draws. A low inefficiency factor indicates better mixing
#' and an efficient algorithm.
#'
#' @return Returns a column vector of inefficiency factors for each component of \eqn{\beta} and \eqn{\sigma}.
#'
#' @importFrom "pracma" "Reshape" "std"
#' @importFrom "stats" "acf"
#'
#' @references Greenberg, E. (2012). `"Introduction to Bayesian Econometrics."` Cambridge University
#' Press, Cambridge. DOI: 10.1017/CBO9780511808920
#'
#' Chib, S. (2012), `"Introduction to simulation and MCMC methods."` In Geweke J., Koop G., and Dijk, H.V.,
#' editors, `"The Oxford Handbook of Bayesian Econometrics"`, pages 183--218. Oxford University Press,
#' Oxford. DOI: 10.1093/oxfordhb/9780199559084.013.0006
#'
#' @seealso pracma, \link[stats]{acf}
#' @examples
#' set.seed(101)
#' data("data25j3")
#' y <- data25j3$y
#' xMat <- data25j3$x
#' k <- dim(xMat)[2]
#' b0 <- array(rep(0, k), dim = c(k, 1))
#' B0 <- 10*diag(k)
#' n0 <- 5
#' d0 <- 8
#' output <- quantregOR2(y = y, x = xMat, b0, B0, n0, d0, gammacp2 = 3,
#' burn = 10, mcmc = 40, p = 0.25, accutoff = 0.5, maxlags = 400, verbose = FALSE)
#' betadraws <- output$betadraws
#' sigmadraws <- output$sigmadraws
#'
#' inefficiency <- ineffactorOR2(xMat, betadraws, sigmadraws, 0.5, 400, TRUE)
#'
#' # Summary of Inefficiency Factor:
#' #            Inef Factor
#' # beta_1       1.5686
#' # beta_2       1.5240
#' # beta_3       1.4807
#' # sigma        2.4228
#'
#' @export
ineffactorOR2 <- function(x, betadraws, sigmadraws, accutoff = 0.05, maxlags = 400, verbose = TRUE) {
    cols <- colnames(x)
    names(x) <- NULL
    x <- as.matrix(x)
    betadraws <- as.matrix(betadraws)
    sigmadraws <- as.matrix(sigmadraws)
    if ( !all(is.numeric(betadraws))){
        stop("each entry in betadraws must be numeric")
    }
    n <- dim(betadraws)[2]
    k <- dim(betadraws)[1]
    inefficiencyBeta <- array(0, dim = c(k, 1))
    for (i in 1:k) {
        autocorrelation <- acf(betadraws[i,], lag.max = maxlags, plot = FALSE)
        nlags <- tryCatch(min(which(autocorrelation$acf <= accutoff)), warning = function(w){
            message("Increase either the maxlags or accutoff \n", w)})
        nbatch <- floor(n / nlags)
        nuse <- nbatch * nlags
        b <- betadraws[i, 1:nuse]
        xbatch <- Reshape(b, nlags, nbatch)
        mxbatch <- colMeans(xbatch)
        varxbatch <- sum( (t(mxbatch) - mean(b)) *
                              (t(mxbatch) - mean(b))) / (nbatch - 1)
        nse <- sqrt(varxbatch / (nbatch))
        rne <- (std(b, 1) / sqrt( nuse )) / nse
        inefficiencyBeta[i, 1] <- 1 / rne
    }
    if ( !all(is.numeric(sigmadraws))){
        stop("each entry in sigmadraws must be numeric")
    }
    inefficiencySigma <- array(0, dim = c(1))
    autocorrelation <- acf(c(sigmadraws), lag.max = maxlags, plot = FALSE)
    nlags <- tryCatch(min(which(autocorrelation$acf <= accutoff)), warning = function(w){
        message("Increase either the maxlags or accutoff \n", w)})
    nbatch2 <- floor(n / nlags)
    nuse2 <- nbatch2 * nlags
    b2 <- sigmadraws[1:nuse2]
    xbatch2 <- Reshape(b2, nlags, nbatch2)
    mxbatch2 <- colMeans(xbatch2)
    varxbatch2 <- sum( (t(mxbatch2) - mean(b2)) *
                               (t(mxbatch2) - mean(b2))) / (nbatch2 - 1)
    nse2 <- sqrt(varxbatch2 / (nbatch2))
    rne2 <- (std(b2, 1) / sqrt( nuse2 )) / nse2
    inefficiencySigma <- 1 / rne2

    inefficiencyRes <- rbind(inefficiencyBeta, inefficiencySigma)
    name <- list('Inef Factor')
    dimnames(inefficiencyRes)[[2]] <- name
    dimnames(inefficiencyRes)[[1]] <- letters[1:(k+1)]
    j <- 1
    if (is.null(cols)) {
        rownames(inefficiencyRes)[j] <- c('Intercept')
        for (i in paste0("beta_",1:k)) {
            rownames(inefficiencyRes)[j] = i
            j = j + 1
        }
    }
    else {
        for (i in cols) {
            rownames(inefficiencyRes)[j] = i
            j = j + 1
        }
    }
    rownames(inefficiencyRes)[j] <- 'sigma'
    if(verbose) {
    print(noquote('Summary of Inefficiency Factor: '))
    cat("\n")
    print(round(inefficiencyRes, 4))
    }

    return(inefficiencyRes)
}
#' Covariate effect in the OR2 model
#'
#' This function computes the average covariate effect for different
#' outcomes of the OR2 model at a specified quantile. The covariate
#' effects are calculated marginally of the parameters and the remaining covariates.
#'
#' @usage covEffectOR2(modelOR2, y, xMat1, xMat2, gammacp2, p, verbose)
#'
#' @param modelOR2  output from the quantregOR2 function.
#' @param y         observed ordinal outcomes, column vector of size \eqn{(n x 1)}.
#' @param xMat1     covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#'                  If the covariate of interest is continuous, then the column for the covariate of interest remains unchanged.
#'                  If it is an indicator variable then replace the column for the covariate of interest with a
#'                  column of zeros.
#' @param xMat2     covariate matrix x with suitable modification to an independent variable including a column of ones with
#'                  or without column names. If the covariate of interest is continuous, then add the incremental change
#'                  to each observation in the column for the covariate of interest. If the covariate is an indicator variable,
#'                  then replace the column for the covariate of interest with a column of ones.
#' @param gammacp2    one and only cut-point other than 0.
#' @param p         quantile level or skewness parameter, p in (0,1).
#' @param verbose   whether to print the final output and provide additional information or not, default is TRUE.
#'
#' @details
#' This function computes the average covariate effect for different
#' outcomes of the OR2 model at a specified quantile. The covariate
#' effects are computed, using the Gibbs draws, marginally of the parameters
#' and the remaining covariates.
#'
#' @return Returns a list with components:
#' \item{\code{avgDiffProb}: }{vector with change in predicted
#' probability for each outcome category.}
#'
#' @references Rahman, M. A. (2016). `"Bayesian Quantile Regression for Ordinal Models."`
#' Bayesian Analysis, 11(1): 1-24. DOI: 10.1214/15-BA939
#'
#' Jeliazkov, I., Graves, J., and Kutzbach, M. (2008). `"Fitting and Comparison of Models for Multivariate Ordinal Outcomes."`
#' Advances in Econometrics: Bayesian Econometrics, 23: 115`-`156. DOI: 10.1016/S0731-9053(08)23004-5
#'
#' Jeliazkov, I., and Rahman, M. A. (2012). `"Binary and Ordinal Data Analysis in Economics: Modeling and Estimation"`
#' in Mathematical Modeling with Multidisciplinary
#' Applications, edited by X.S. Yang, 123-150. John Wiley `&` Sons Inc, Hoboken, New Jersey. DOI: 10.1002/9781118462706.ch6
#'
#' @importFrom "stats" "sd"
#' @examples
#' set.seed(101)
#' data("data25j3")
#' y <- data25j3$y
#' xMat1 <- data25j3$x
#' k <- dim(xMat1)[2]
#' b0 <- array(rep(0, k), dim = c(k, 1))
#' B0 <- 10*diag(k)
#' n0 <- 5
#' d0 <- 8
#' output <- quantregOR2(y, xMat1, b0, B0, n0, d0, gammacp2 = 3,
#' burn = 10, mcmc = 40, p = 0.25, accutoff = 0.5, maxlags = 400, verbose = FALSE)
#' xMat2 <- xMat1
#' xMat2[,3] <- xMat2[,3] + 0.02
#' res <- covEffectOR2(output, y, xMat1, xMat2, gammacp2 = 3, p = 0.25, verbose = TRUE)
#'
#' # Summary of Covariate Effect:
#'
#' #               Covariate Effect
#' # Category_1          -0.0073
#' # Category_2          -0.0030
#' # Category_3           0.0103
#'
#' @export
covEffectOR2 <- function(modelOR2, y, xMat1, xMat2, gammacp2, p, verbose = TRUE) {
    cols <- colnames(xMat1)
    cols1 <- colnames(xMat2)
    names(xMat1) <- NULL
    names(y) <- NULL
    names(xMat2) <- NULL
    xMat1 <- as.matrix(xMat1)
    xMat2 <- as.matrix(xMat2)
    y <- as.matrix(y)
    J <- dim(as.array(unique(y)))[1]
    if ( J > 3 ){
        stop("This function is only available for models with 3 outcome
                variables.")
    }
    if (dim(y)[2] != 1){
        stop("input y should be a column vector")
    }
    if ( any(!all(y == floor(y)))){
        stop("each entry of y must be an integer")
    }
    if ( !all(is.numeric(xMat1))){
        stop("each entry in xMat1 must be numeric")
    }
    if ( !all(is.numeric(xMat2))){
        stop("each entry in xMat2 must be numeric")
    }
    if ( length(p) != 1){
        stop("parameter p must be scalar")
    }
    if ( length(gammacp2) != 1){
        stop("parameter gammacp2 must be scalar")
    }
    if ( !is.numeric(gammacp2)){
        stop("parameter gammacp2 must be a numeric")
    }
    if (any(p < 0 | p > 1)){
        stop("parameter p must be between 0 to 1")
    }
    N <- dim(modelOR2$betadraws)[2]
    m <- (N)/(1.25)
    burn <- 0.25 * m
    n <- dim(xMat1)[1]
    k <- dim(xMat1)[2]
    betaBurnt <- modelOR2$betadraws[, (burn + 1):N]
    sigmaBurnt <- modelOR2$sigmadraws[(burn + 1):N]
    betaBurnt <- as.matrix(betaBurnt)
    sigmaBurnt <- as.matrix(sigmaBurnt)
    mu <- 0
    gammacp <- array(c(-Inf, 0, gammacp2, Inf), dim = c(1, J))
    oldProb <- array(0, dim = c(n, m, J))
    newProb <- array(0, dim = c(n, m, J))
    oldComp <- array(0, dim = c(n, m, (J-1)))
    newComp <- array(0, dim = c(n, m, (J-1)))

    if(verbose) {
        pb <- progress_bar$new(" Computation of CE in Progress [:bar] :percent",
                               total = (J-1)*(m)*(n), clear = FALSE, width = 100)
    }

    for (j in 1:(J-1)) {
        for (b in 1:m) {
            for (i in 1:n) {
                oldComp[i, b, j] <- alcdf((gammacp[j+1] - (xMat1[i, ] %*% betaBurnt[, b])), mu, sigmaBurnt[b], p)
                newComp[i, b, j] <- alcdf((gammacp[j+1] - (xMat2[i, ] %*% betaBurnt[, b])), mu, sigmaBurnt[b], p)
            }
            if (j == 1) {
                oldProb[, b, j] <- oldComp[, b, j]
                newProb[, b, j] <- newComp[, b, j]
            }
            else {
                oldProb[, b, j] <- oldComp[, b, j] - oldComp[, b, (j-1)]
                newProb[, b, j] <- newComp[, b, j] - newComp[, b, (j-1)]
            }
        }
    }
    oldProb[, , J] = 1 - oldComp[, , (J-1)]
    newProb[, , J] = 1 - newComp[, , (J-1)]
    diffProb <- newProb - oldProb
    avgDiffProb <- array((colMeans(diffProb, dims = 2)), dim = c(J, 1))
    name <- list('Covariate Effect')
    dimnames(avgDiffProb)[[2]] <- name
    dimnames(avgDiffProb)[[1]] <- letters[1:(J)]
    ordOutput <- as.array(unique(y))
    j <- 1
    for (i in paste0("Category_",1:J)) {
        rownames(avgDiffProb)[j] = i
        j = j + 1
    }
    if(verbose) {
    print(noquote('Summary of Covariate Effect: '))
    cat("\n")
    print(round(avgDiffProb, 4))
    }

    result <- list("avgDiffProb" = avgDiffProb)

    return(result)
}
#' Marginal likelihood in the OR2 model
#'
#' This function computes the logarithm of marginal likelihood in the OR2 model (ordinal
#' quantile model with exactly 3 outcomes) using the Gibbs output from the
#' complete and reduced runs.
#'
#' @usage logMargLikeOR2(y, x, b0, B0, n0, d0, postMeanbeta, postMeansigma,
#' btildeStore, BtildeStore, gammacp2, p, verbose)
#'
#' @param y                 observed ordinal outcomes, column vector of size \eqn{(n x 1)}.
#' @param x                 covariate matrix of size \eqn{(n x k)} including a column of ones with or without column names.
#' @param b0                prior mean for \eqn{\beta}.
#' @param B0                prior covariance matrix for \eqn{\beta}.
#' @param n0                prior shape parameter of inverse-gamma distribution for \eqn{\sigma}.
#' @param d0                prior scale parameter of inverse-gamma distribution for \eqn{\sigma}.
#' @param postMeanbeta      posterior mean of \eqn{\beta} from the complete Gibbs run.
#' @param postMeansigma     posterior mean of \eqn{\delta} from the complete Gibbs run.
#' @param btildeStore       a storage matrix for btilde from the complete Gibbs run.
#' @param BtildeStore       a storage matrix for Btilde from the complete Gibbs run.
#' @param gammacp2            one and only cut-point other than 0.
#' @param p                 quantile level or skewness parameter, p in (0,1).
#' @param verbose           whether to print the final output and provide additional information or not, default is TRUE.
#'
#' @details
#' This function computes the logarithm of marginal likelihood in the OR2 model using the Gibbs output from the complete
#' and reduced runs.
#'
#' @return Returns an estimate of log marginal likelihood
#'
#' @references Chib, S. (1995). `"Marginal likelihood from the Gibbs output."` Journal of the American
#' Statistical Association, 90(432):1313`-`1321, 1995. DOI: 10.1080/01621459.1995.10476635
#'
#' @importFrom "stats" "sd" "dnorm"
#' @importFrom "invgamma" "dinvgamma"
#' @importFrom "pracma" "inv"
#' @importFrom "NPflow" "mvnpdf"
#' @importFrom "progress" "progress_bar"
#' @seealso \link[invgamma]{dinvgamma}, \link[NPflow]{mvnpdf}, \link[stats]{dnorm},
#' Gibbs sampling
#' @examples
#' set.seed(101)
#' data("data25j3")
#' y <- data25j3$y
#' xMat <- data25j3$x
#' k <- dim(xMat)[2]
#' b0 <- array(rep(0, k), dim = c(k, 1))
#' B0 <- 10*diag(k)
#' n0 <- 5
#' d0 <- 8
#' output <- quantregOR2(y = y, x = xMat, b0, B0, n0, d0, gammacp2 = 3,
#' burn = 10, mcmc = 40, p = 0.25, accutoff = 0.5, maxlags = 400, verbose = FALSE)
#' # output$logMargLike
#' #   -404.57
#'
#' @export
logMargLikeOR2 <- function(y, x, b0, B0, n0, d0, postMeanbeta, postMeansigma, btildeStore, BtildeStore, gammacp2, p, verbose) {
    cols <- colnames(x)
    names(x) <- NULL
    names(y) <- NULL
    x <- as.matrix(x)
    y <- as.matrix(y)
    if ( dim(y)[2] != 1){
        stop("input y should be a column vector")
    }
    if ( any(!all(y == floor(y)))){
        stop("each entry of y must be an integer")
    }
    if ( !all(is.numeric(x))){
        stop("each entry in x must be numeric")
    }
    if ( length(p) != 1){
        stop("parameter p must be scalar")
    }
    if ( any(p < 0 | p > 1)){
        stop("parameter p must be between 0 to 1")
    }
    if ( !all(is.numeric(b0))){
        stop("each entry in b0 must be numeric")
    }
    if ( length(n0) != 1){
        stop("parameter n0 must be scalar")
    }
    if ( !all(is.numeric(n0))){
        stop("parameter n0 must be numeric")
    }
    if ( length(d0) != 1){
        stop("parameter d0 must be scalar")
    }
    if ( !all(is.numeric(d0))){
        stop("parameter d0 must be numeric")
    }
    J <- dim(as.array(unique(y)))[1]
    if ( J > 3 ){
        stop("This function is for 3 outcome
                variables. Please correctly specify the inputs
             to use quantregOR2")
    }
    n <- dim(x)[1]
    k <- dim(x)[2]
    nsim <- dim(btildeStore)[2]
    burn <- (0.25 * nsim) / (1.25)
    nu <- array(5 * rep(1,n), dim = c(n, 1))
    ntilde <- n0 + (3 * n)
    gammacp <- array(c(-Inf, 0, gammacp2, Inf), dim = c(1, J+1))
    indexp <- 0.5
    theta <- (1 - 2 * p) / (p * (1 - p))
    tau <- sqrt(2 / (p * (1 - p)))
    tau2 <- tau^2
    sigmaRedrun <- array(0, dim = c(1, nsim))
    dtildeStoreRedrun <- array(0, dim = c(1, nsim))
    z <- array( (rnorm(n, mean = 0, sd = 1)), dim = c(n, 1))
    b0 <- array(rep(b0, k), dim = c(k, 1))
    j <- 1
    postOrdbetaStore <- array(0, dim=c((nsim-burn),1))
    postOrdsigmaStore <- array(0, dim=c((nsim-burn),1))
    if(verbose) {
        pb <- progress_bar$new(" Reduced Run in Progress [:bar] :percent",
                           total = nsim, clear = FALSE, width = 100)
    }

    for (i in 1:nsim) {
        sigmaStoreRedrun <- drawsigmaOR2(z, x, postMeanbeta, nu, tau2, theta, n0, d0)
        sigmaRedrun[i] <- sigmaStoreRedrun$sigma
        dtildeStoreRedrun[i] <- sigmaStoreRedrun$dtilde

        nu <- drawnuOR2(z, x, postMeanbeta, sigmaRedrun[i], tau2, theta, indexp)

        z <- drawlatentOR2(y, x, postMeanbeta, sigmaRedrun[i], nu, theta, tau2, gammacp)
        if(verbose) {
            pb$tick()
        }
    }

    sigmaStar <- postMeansigma
    if(verbose) {
        pb <- progress_bar$new(" Calculating Marginal Likelihood [:bar] :percent",
                           total = (nsim-burn), clear = FALSE, width = 100)
    }

    for (i in (burn+1):(nsim)) {
        postOrdbetaStore[j] <- mvnpdf(x = matrix(postMeanbeta), mean = btildeStore[, i], varcovM = BtildeStore[, , i], Log = FALSE)
        postOrdsigmaStore[j] <- (dinvgamma(sigmaStar, shape = (ntilde / 2), scale = (2 / dtildeStoreRedrun[i])))
        j <- j  + 1
        if(verbose) {
            pb$tick()
        }
    }
    postOrdbeta <- mean(postOrdbetaStore)
    postOrdsigma <- mean(postOrdsigmaStore)

    priorContbeta <- mvnpdf(matrix(postMeanbeta), mean = b0, varcovM = B0, Log = FALSE)
    priorContsigma <- dinvgamma(postMeansigma, shape = (n0 / 2), scale = (2 / d0))

    logLikeCont <- -1 * qrnegLogLikeOR2(y, x, gammacp, postMeanbeta, postMeansigma, p)
    logPriorCont <- log(priorContbeta*priorContsigma)
    logPosteriorCont <- log(postOrdbeta*postOrdsigma)

    logMargLike <- logLikeCont + logPriorCont - logPosteriorCont
    return(logMargLike)
}

#' Extractor function for summary
#'
#' This function extracts the summary from the bqrorOR2 object
#'
#' @usage \method{summary}{bqrorOR2}(object, digits, ...)
#'
#' @param object    bqrorOR2 object from which the summary is extracted.
#' @param digits    controls the number of digits after the decimal
#' @param ...       extra arguments
#'
#' @details
#' This function is an extractor function for the summary
#'
#' @return the summarized information object
#'
#' @examples
#' set.seed(101)
#' data("data25j3")
#' y <- data25j3$y
#' xMat <- data25j3$x
#' k <- dim(xMat)[2]
#' b0 <- array(rep(0, k), dim = c(k, 1))
#' B0 <- 10*diag(k)
#' n0 <- 5
#' d0 <- 8
#' output <- quantregOR2(y = y, x = xMat, b0, B0, n0, d0, gammacp2 = 3,
#' burn = 10, mcmc = 40, p = 0.25, accutoff = 0.5, maxlags = 400, FALSE)
#' summary(output, 4)
#'
#' #            Post Mean Post Std Upper Credible Lower Credible Inef Factor
#' #    beta_1   -4.5185   0.9837        -3.1726        -6.2000     1.5686
#' #    beta_2    6.1825   0.9166         7.6179         4.8619     1.5240
#' #    beta_3    5.2984   0.9653         6.9954         4.1619     1.4807
#' #    sigma     1.0879   0.2073         1.5670         0.8436     2.4228
#'
#' @exportS3Method summary bqrorOR2
summary.bqrorOR2 <- function(object, digits = 4,...)
{
    print(round(object$summary, digits))
}
